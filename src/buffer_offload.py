"""Host buffer manager for ELO meta-training.

Manages buffer_inner_opt_states entirely in CPU host memory, eliminating
GPU memory overhead from the JIT scan carry state. Buffer read/write
happens at Python level between scan windows.
"""
import numpy as np
import jax


class HostBufferManager:
    """Manages ELO buffer states in CPU host memory.

    Instead of keeping buffer_inner_opt_states on GPU as part of the JIT scan
    carry state (which consumes ~50% of per-task GPU memory), this class:
    - Stores buffer entries as numpy pytrees in CPU RAM
    - Handles buffer resets (read) before each scan window
    - Handles buffer pushes (write) after each scan window
    - Tracks buffer_cfg (thred, update_idx, idx2push) in Python
    """

    def __init__(self, buffer_cfg, num_tasks):
        self.buffer_size = int(buffer_cfg.get('buffer_size', 1))
        self.thred = float(buffer_cfg.get('thred', 0.3))
        self.min_thred = float(buffer_cfg.get('min_thred', 0.3))
        self.num_tasks = num_tasks

        # Per-task CPU buffers: [num_tasks][buffer_size] of numpy pytrees
        self._pos_buffers = None
        self._neg_buffers = None
        # Per-task inner_step at buffer save time: [num_tasks][buffer_size]
        self._inner_steps = None
        # Per-task tracking (numpy, shape [num_tasks])
        self._update_idx = None
        self._idx2push = None
        self._initialized = False

    def _ensure_initialized(self, p_state, n_state, key):
        """Auto-initialize from current state if not yet initialized.

        Called on first pre_window. Also handles checkpoint resume:
        after loading from checkpoint, the buffer_manager is freshly created
        and _initialized=False, so it re-initializes from the loaded state.
        """
        if self._initialized:
            return

        self._pos_buffers = []
        self._neg_buffers = []
        self._inner_steps = []
        self._task_params = []

        inner_steps = np.array(jax.device_get(p_state.inner_step))
        for i in range(self.num_tasks):
            p_opt = jax.tree_util.tree_map(lambda x: x[i], p_state.inner_opt_state)
            n_opt = jax.tree_util.tree_map(lambda x: x[i], n_state.inner_opt_state)
            tp = jax.tree_util.tree_map(lambda x: x[i], p_state.task_param)
            self._pos_buffers.append([jax.device_get(p_opt)] * self.buffer_size)
            self._neg_buffers.append([jax.device_get(n_opt)] * self.buffer_size)
            self._inner_steps.append([int(inner_steps[i])] * self.buffer_size)
            self._task_params.append([jax.device_get(tp)] * self.buffer_size)

        self._update_idx = np.zeros(self.num_tasks, dtype=np.int32)

        trunc_lengths = np.array(jax.device_get(p_state.truncation_state.length))
        inner_steps = np.array(jax.device_get(p_state.inner_step))
        seed = int(jax.device_get(key)[0]) % (2**31)
        rng = np.random.RandomState(seed)
        self._idx2push = np.array([
            rng.randint(int(inner_steps[i]),
                        max(int(trunc_lengths[i]), int(inner_steps[i]) + 1))
            for i in range(self.num_tasks)
        ], dtype=np.int32)

        self._initialized = True

    def update_thred(self, outer_iteration):
        """Decay buffer threshold based on outer iteration."""
        outer_iter = float(jax.device_get(outer_iteration))
        self.thred = 0.3 - (0.3 - self.min_thred) * min(1.0, outer_iter / 3000)

    def pre_window(self, p_state, n_state, key):
        """Prepare buffer candidates for all tasks before scan window.

        For each task, pre-sample the buffer/random decision and load buffer
        candidate onto GPU. The actual reset happens inside the JIT scan via
        use_buffer_on_reset flag in the state.

        Returns:
            Updated (p_state, n_state) with pending_buffer fields populated.
        """
        self._ensure_initialized(p_state, n_state, key)

        device = jax.local_devices()[0]
        seed = int(jax.device_get(key)[0]) % (2**31)
        rng = np.random.RandomState(seed)

        # Prepare buffer candidates for all tasks (batch)
        all_pos_buf_opts = []
        all_neg_buf_opts = []
        all_buf_tps = []
        all_buf_steps = []
        all_use_buffer = []

        for i in range(self.num_tasks):
            buffer_prob = rng.uniform()
            use_buffer = buffer_prob > self.thred

            if use_buffer:
                select_idx = rng.randint(0, self.buffer_size)
                all_pos_buf_opts.append(self._pos_buffers[i][select_idx])
                all_neg_buf_opts.append(self._neg_buffers[i][select_idx])
                all_buf_tps.append(self._task_params[i][select_idx])
                all_buf_steps.append(self._inner_steps[i][select_idx])
                all_use_buffer.append(True)
            else:
                # Fill with zeros as placeholder (same tree structure)
                all_pos_buf_opts.append(
                    jax.tree_util.tree_map(np.zeros_like, self._pos_buffers[i][0]))
                all_neg_buf_opts.append(
                    jax.tree_util.tree_map(np.zeros_like, self._neg_buffers[i][0]))
                all_buf_tps.append(
                    jax.tree_util.tree_map(np.zeros_like, self._task_params[i][0]))
                all_buf_steps.append(0)
                all_use_buffer.append(False)

        # Stack into vmapped arrays and device_put in one batch
        stacked_pos_opts = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs), *all_pos_buf_opts)
        stacked_neg_opts = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs), *all_neg_buf_opts)
        stacked_tps = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs), *all_buf_tps)
        stacked_steps = np.array(all_buf_steps, dtype=np.int32)
        stacked_use_buffer = np.array(all_use_buffer)

        p_state = p_state.replace(
            pending_buffer_opt_state=jax.device_put(stacked_pos_opts, device),
            pending_buffer_task_param=jax.device_put(stacked_tps, device),
            pending_buffer_inner_step=jax.device_put(stacked_steps, device),
            use_buffer_on_reset=jax.device_put(stacked_use_buffer, device),
        )
        n_state = n_state.replace(
            pending_buffer_opt_state=jax.device_put(stacked_neg_opts, device),
            pending_buffer_task_param=jax.device_put(stacked_tps, device),
            pending_buffer_inner_step=jax.device_put(stacked_steps, device),
            use_buffer_on_reset=jax.device_put(stacked_use_buffer, device),
        )

        return p_state, n_state

    def post_window(self, p_state, n_state, prev_p_steps):
        """Handle buffer pushes and idx2push updates after scan window.

        1. For tasks that crossed idx2push: push opt state to CPU buffer.
        2. For tasks that reset mid-window (step went backwards): update idx2push
           for the new trajectory.

        Args:
            p_state, n_state: states after scan window
            prev_p_steps: numpy array of inner_step values before window
        """
        if not self._initialized:
            return

        curr_p_steps = np.array(jax.device_get(p_state.inner_step))
        seed = int(np.sum(curr_p_steps)) % (2**31)
        rng = np.random.RandomState(seed)

        for i in range(self.num_tasks):
            if curr_p_steps[i] < prev_p_steps[i]:
                # Task was reset mid-window (step went backwards)
                # Update idx2push for the new trajectory
                trunc_length = int(jax.device_get(p_state.truncation_state.length[i]))
                inner_step = int(curr_p_steps[i])
                if trunc_length > inner_step:
                    self._idx2push[i] = rng.randint(inner_step, trunc_length)
                else:
                    self._idx2push[i] = inner_step
                continue

            # Check if idx2push was crossed during this window
            if prev_p_steps[i] <= self._idx2push[i] < curr_p_steps[i]:
                # print(f"[Buffer] Task {i}: push_to_buffer at step={int(curr_p_steps[i])}, idx2push={int(self._idx2push[i])}, slot={int(self._update_idx[i]) % self.buffer_size}")
                p_opt = jax.tree_util.tree_map(lambda x: x[i], p_state.inner_opt_state)
                n_opt = jax.tree_util.tree_map(lambda x: x[i], n_state.inner_opt_state)
                tp = jax.tree_util.tree_map(lambda x: x[i], p_state.task_param)

                idx = int(self._update_idx[i]) % self.buffer_size
                self._pos_buffers[i][idx] = jax.device_get(p_opt)
                self._neg_buffers[i][idx] = jax.device_get(n_opt)
                self._inner_steps[i][idx] = int(curr_p_steps[i])
                self._task_params[i][idx] = jax.device_get(tp)
                self._update_idx[i] = (self._update_idx[i] + 1) % self.buffer_size
