import numpy as np

def dynamics(beta, J, h, sigma, dynamic, dyn_rng):

    try:
        layers, neurons = np.shape(sigma)
        noise = dyn_rng.uniform(low = -1, high = 1, size = (layers, neurons))

        if dynamic == 'parallel':
            if np.isinf(beta):
                return np.sign(np.einsum('klij,lj->ki', J, sigma) + h)
            else:
                return np.sign(np.tanh(beta * (np.einsum('klij,lj->ki', J, sigma) + h)) + noise)
        elif dynamic == 'sequential':
            new_sigma = sigma.copy()
            neuron_sampling = dyn_rng.permutation(range(neurons))
            for idx_N in neuron_sampling:
                layer_sampling = dyn_rng.permutation(range(layers))
                for idx_L in layer_sampling:
                    if np.isinf(beta):
                        new_neuron = np.sign(np.einsum('ki, ki -> ', J[idx_L, :, idx_N, :], new_sigma)
                                            + h[idx_L, idx_N])
                    else:
                        new_neuron = np.sign(
                        np.tanh(beta * (np.einsum('ki, ki -> ', J[idx_L,:,idx_N,:], new_sigma)
                                        + h[idx_L, idx_N])) + noise[idx_L, idx_N])
                    new_sigma[idx_L, idx_N] = new_neuron
            return new_sigma
        else:
            raise Exception('No valid dynamic update rule given.')

    except ValueError:

        if dynamic == 'parallel':
            if np.isinf(beta):
                return np.sign(sigma@J + h)
            else:
                noise = dyn_rng.uniform(low=-1, high=1, size=np.shape(sigma))
                return np.sign(np.tanh(beta * (sigma@J + h)) + noise)

        elif dynamic == 'sequential':
            new_sigma = sigma.copy()
            neuron_sampling = dyn_rng.permutation(range(np.shape(sigma)[-1]))
            if np.isinf(beta):
                for idx_N in neuron_sampling:
                    new_neuron = np.sign(new_sigma@J[...,idx_N] + h[...,idx_N])
                    new_sigma[..., idx_N] = new_neuron
            else:
                for idx_N in neuron_sampling:
                    noise = dyn_rng.uniform(low=-1, high=1, size=np.shape(sigma))
                    new_neuron = np.sign(np.tanh(beta * ( new_sigma@J[...,idx_N] + h[...,idx_N])) + noise[...,idx_N])
                    new_sigma[...,idx_N] = new_neuron
            return new_sigma
        else:
            raise Exception('No valid dynamic update rule given.')


def dynamics_simple_noh(beta, J, sigma, dynamic, dyn_rng):

    if dynamic == 'parallel':
        if np.isinf(beta):
            return np.sign(sigma @ J)
        else:
            noise = dyn_rng.uniform(low=-1, high=1, size=np.shape(sigma))
            return np.sign(np.tanh(beta * (sigma @ J)) + noise)

    elif dynamic == 'sequential':
        new_sigma = sigma.copy()
        neuron_sampling = dyn_rng.permutation(range(np.shape(sigma)[-1]))
        if np.isinf(beta):
            for idx_N in neuron_sampling:
                new_neuron = np.sign(new_sigma @ J[..., idx_N])
                new_sigma[..., idx_N] = new_neuron
        else:
            for idx_N in neuron_sampling:
                noise = dyn_rng.uniform(low=-1, high=1, size=np.shape(sigma))
                new_neuron = np.sign(np.tanh(beta * (new_sigma @ J[..., idx_N])) + noise[..., idx_N])
                new_sigma[..., idx_N] = new_neuron
        return new_sigma
    else:
        raise Exception('No valid dynamic update rule given.')

def g(layers, lmb):
    matrix_g = np.full((layers, layers), -lmb)
    np.fill_diagonal(matrix_g, 1)
    return matrix_g


class TAM:

    def __init__(self, neurons, layers, split, supervised, r, m, k = 0, patterns = None, lmb = -1, rng_ss = np.random.SeedSequence()):

        # usage of SeedSequence objects allows for reproducibility
        # create one seed sequence for each independent source of randomness
        rng_seeds = rng_ss.spawn(4)

        # fast noise uses a seed sequence, since simulate always starts from the initial state
        # in order to get independent runs of the same system, one should further spawn independent seeds from this
        # simulate does this by default, but one should keep it in mind nonetheless
        self.fast_noise = rng_seeds[0]
        # slow noise already uses bit generators since the method add_patterns always starts from where it left off
        # if we used a seed sequence, it would start from the beginning everytime we called methods like add_pattern
        # for reproducibility, one should always keep in mind the different seeds and where each one "starts"
        self.noise_patterns = np.random.default_rng(rng_seeds[1])
        self.noise_examples = np.random.default_rng(rng_seeds[2])
        self.noise_initial = rng_seeds[3]

        self._neurons = neurons
        self._layers = layers

        self._r = r
        self._m = m

        # we set k at 0 first because the pattern setter can only be called for k = 0
        self._k = 0

        # initializes the patterns (see patterns setter)
        # it is used when we want to give specific patterns and not randomly generate them
        # for instance when copying them from another system
        # after this we are only allowed to add more patterns by increasing k and they are randomly generated
        # examples for each pattern are randomly generated (see patterns setter)
        if patterns is not None:
            self.patterns = patterns
        else:
            self._k = 0
            self._patterns = self.gen_patterns(0)

        # available examples
        self._examples = self.gen_examples(self._patterns)

        # variables for the computation of the interaction matrix
        self._lmb = lmb
        self._split = split
        self._supervised = supervised

        # interaction matrix and effective examples
        # effective examples are the examples above but split among layers (if split) and average among examples (if supervised)
        # gets defined when interaction matrix does (see set_interaction)
        self._J = None
        self._effective_examples = None


        # next we add random patterns and examples
        # can only be higher than existing patterns
        if k > self._k:
            self.add_patterns(k - self.k) # this one already constructs interaction matrix by itself
        elif k > 0:
            # interaction matrix gets constructed with the set_interaction method
            self.set_interaction()
        if k < self._k:
            print('Invalid k input will be ignored.')

        # the initial state and external field to be used in simulate
        # defined manually outside the constructor
        # can use the methods mix, dis, etc (define them as needed)
        self.external_field = None
        self.initial_state = None

    @property
    def lmb(self):
        return self._lmb

    @property
    def neurons(self):
        return self._neurons

    @property
    def layers(self):
        return self._layers

    @property
    def r(self):
        return self._r

    @property
    def m(self):
        return self._m

    @property
    def J(self):
        return self._J

    @property
    def patterns(self):
        return self._patterns

    @property
    def examples(self):
        return self._examples

    @property
    def k(self):
        return self._k

    @property
    def m_per_layer(self):
        if self._split:
            if self._m % self._layers != 0:
                print('Warning: non-integer m per layer.')
            return self.m // self._layers
        else:
            return self._m


    @patterns.setter
    def patterns(self, patterns):
        assert self._k == 0, 'Patterns have already been set. Use add_patterns instead.'
        assert isinstance(patterns, np.ndarray) and len(patterns.shape) == 2 and patterns.shape[1] == self._neurons, 'Invalid pattern input.'
        self._k = patterns.shape[0]
        self._patterns = patterns
        self._examples = self.gen_examples(self._patterns)

    def add_patterns(self, k):
        assert isinstance(k, int) and k > 0, 'Number of patterns can only be increased.'
        extra_patterns = self.gen_patterns(k)
        extra_examples = self.gen_examples(extra_patterns)
        self._patterns = np.concatenate((self._patterns, extra_patterns))
        self._examples = np.concatenate((self._examples, extra_examples), axis=1)

        if self._J is not None:
            self._J = self._J + self.interaction(extra_examples)
            self._effective_examples = self.effective_examples()
        else:
            self.set_interaction()
        self._k += k


    # constructor of the interaction matrix (also used when patterns are added)
    # for cases where all layers are the same (ie rho = 0 or not-split), and lambda is not given, this is a neurons * neurons matrix
    # this way, one can save memory and reuse the same system for different lambda's
    # otherwise it has dimensions layers * layers * neurons * neurons with lambda = -1
    # simulate allows for a matrix J input because of this
    # and the method insert_g allows us to insert the lambda dependence latter

    # the reason this method and set_interaction are not the same method is for this one to be used for the extra patterns in add_patterns
    def interaction(self, examples = None):
        if examples is None:
            examples = self._examples
        big_r = self._r ** 2 + (1 - self._r ** 2) / self.m_per_layer
        if self._lmb >= 0 or self._split: # in these cases the interaction matrix already has full dimensions
            eff_examples = self.effective_examples(examples)
            if self._supervised:
                J = (1 / (big_r * self.neurons)) * np.einsum('kl, kui, luj -> klij', self.g(self._lmb), eff_examples, eff_examples)
            else:
                J = (1 / (big_r * self.neurons * self.m_per_layer)) * np.einsum('kl, kaui, lauj -> klij', self.g(self._lmb), eff_examples,
                                                           eff_examples, optimize = True)
            for i in range(self.neurons):
                for l in range(self.layers):
                    J[l, l, i, i] = 0
        else: # in these cases we keep the interaction matrix with only dimensions neurons * neurons and add the g matrix later
            if self._supervised:
                av_examples = np.mean(examples, axis = 0)
                J = (1 / (big_r * self.neurons)) * np.einsum('ui, uj -> ij', av_examples, av_examples)
            else:
                J = (1 / (big_r * self.neurons * self._m)) * np.einsum('aui, auj -> ij', examples, examples, optimize = True)
        return J

    def effective_examples(self, examples = None):
        if examples is None:
            examples = self._examples
        k = np.shape(examples)[1]
        if self._split:
            applied_examples = np.reshape(examples, (self._layers, self.m_per_layer, k, self._neurons))
        else:
            applied_examples = np.broadcast_to(examples, (self._layers, self._m, k, self._neurons))
        if self._supervised:
            applied_examples = np.mean(applied_examples, axis=1)
        return applied_examples


    def set_interaction(self, lmb = None, split = None, supervised = None):
        if lmb is not None:
            self._lmb = lmb
        if split is not None:
            self._split = split
        if supervised is not None:
            self._supervised = supervised
        self._J = self.interaction()
        self._effective_examples = self.effective_examples()


    def gen_patterns(self, k):
        return self.noise_patterns.choice([-1, 1], (k, self._neurons))

    def gen_examples(self, patterns):
        k = np.shape(patterns)[0]
        blurs = self.noise_examples.choice([-1, 1], p=[(1 - self._r) / 2, (1 + self._r) / 2], size = (k, self._m, self._neurons))
        return np.transpose(blurs, [1,0,2]) * patterns

    # allows us to choose initial states by their names
    def set_initial(self, state = 'mix'):
        if state == 'mix':
            self.initial_state = self.mix()
        elif state == 'dis':
            self.initial_state = self.dis()
        elif state == 'pure':
            self.initial_state = self.pure()
        elif state == 'random':
            self.initial_state = self.random()

    # mixture state
    def mix(self):
        return np.broadcast_to(np.sign(np.sum(self._patterns[:self._layers], axis = 0)), (self._layers, self._neurons))

    # disentangled state
    def dis(self):
        return self._patterns[:self._layers]

    def pure(self, idx = 0):
        return np.broadcast_to(self._patterns[idx], (self._layers, self._neurons))

    def random(self, rng_ss = None):
        if rng_ss is None:
            rng_ss = self.noise_initial.spawn(1)[0]
        return np.random.default_rng(rng_ss).choice([-1,1], (self._layers, self._neurons))

    def zeros(self):
        return np.zeros((self._layers, self._neurons))

    def g(self, lmb):
        return g(layers = self._layers, lmb = lmb)

    def insert_g(self, lmb):
        if len(np.shape(self._J)) == 2:
            J = np.broadcast_to(self._J, (self.layers, self.layers, self.neurons, self.neurons))*self.g(lmb)[:,:,None,None]
            for i in range(self.neurons):
                for l in range(self.layers):
                    J[l, l, i, i] = 0
        else:
            J = self._J*self.g(lmb)[:, :, None, None]
        return J


    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma, cap = None):
        if cap is None:
            cap = self._layers
        return (1 / self._neurons) * np.einsum('li, ui -> lu', sigma, self._patterns[:cap])

    def ex_mags(self, sigma, cap = None):
        if cap is None:
            cap = self._layers
        if self._supervised:
            big_r = self._r ** 2 + (1 - self._r ** 2) / self.m_per_layer
            return (self._r / (self._neurons * big_r)) * np.einsum('li, lui -> lu', sigma, self._effective_examples[:,:cap])
        else:
            # is there a constant here?
            return (1 / self._neurons) * np.einsum('li, aui -> alu', sigma, self._examples[:,:cap])

    # Method simulate runs the MonteCarlo simulation
    # It does L x neurons flips per iteration.
    # Each of these L x neurons flips is one call of the function "dynamics" (defined above)
    # At each iteration it appends the new state a list
    # It loops until a maximum number of iterations is reached
    # Or until the standard deviation in the last av_counter magnetizations is below a certain threshold

    # INPUTS:
    # max_it is the maximum number of iterations
    # beta is the inverse temperature
    # dynamic is either 'parallel' or 'sequential' (see function dynamics)
    # H is the strength of the external field (the external field already exists in self.h)
    # lmb is the value of lambda, in case the interaction matrix does not have it yet
    # error is the threshold for the standard deviation to assert convergence
    # av_counter is the  number of iterations used in the standard deviation / convergence test
    # av = True takes the average of the last av_counter iterations before returning, otherwise it returns the full history

    # It returns the full history of magnetizations
    def simulate(self, beta, max_it, dynamic, error, av_counter, h_norm, sim_J = None, av = True, sim_rng = None):
        assert self.initial_state is not None, 'Initial state not provided.'
        assert self.external_field is not None, 'External field not provided.'

        if av_counter == 1 and error > 0:
            print('Warning: av_counter set to 1 with positive error')

        if sim_rng is None:
            dyn_rng = np.random.default_rng(self.fast_noise.spawn(1)[0])
        else:
            dyn_rng = np.random.default_rng(sim_rng)

        if sim_J is None:
            assert self._lmb >= 0, r'\lambda not available to simulate.'
            sim_J = self._J

        state = self.initial_state

        mags = [self.mattis(state)]
        ex_mags = [self.ex_mags(state)]

        idx = 0
        while idx < max_it: # do the simulation
            idx += 1

            state = dynamics(beta = beta, J = sim_J, h = h_norm * self.external_field, sigma = state, dynamic = dynamic, dyn_rng = dyn_rng)

            mags.append(self.mattis(state))
            ex_mags.append(self.ex_mags(state))
            if idx + 1 >= av_counter: # size of the actual arrays has +1 since they include initial states
                if av_counter > 1:
                    last_error = np.max(np.std(mags[-av_counter:], axis=0))
                else:
                    last_error = np.max(np.abs(mags[-1]-mags[-2]))
                if last_error <= error:
                    break

        if av:
            mags = np.mean(mags[-av_counter:], axis = 0)
            ex_mags = np.mean(ex_mags[-av_counter:], axis=0)

        return mags, ex_mags, idx


class Dream:

    def __init__(self, neurons, r, m, k = 0, t = 0, supervised = False, diagonal = False, rng_ss = None):

        # usage of SeedSequence objects allows for reproducibility
        # create one seed sequence for each independent source of randomness
        if rng_ss is None:
            rng_ss = np.random.SeedSequence()

        rng_seeds = rng_ss.spawn(4)

        # fast noise uses a seed sequence, since simulate always starts from the initial state
        # in order to get independent runs of the same system, one should further spawn independent seeds from this
        # simulate does this by default, but one should keep it in mind nonetheless
        self.fast_noise = rng_seeds[0]
        # slow noise already uses bit generators since the method add_patterns always starts from where it left off
        # if we used a seed sequence, it would start from the beginning everytime we called methods like add_pattern
        # for reproducibility, one should always keep in mind the different seeds and where each one "starts"
        self.noise_patterns = np.random.default_rng(rng_seeds[1])
        self.noise_examples = np.random.default_rng(rng_seeds[2])
        self.noise_initial = rng_seeds[3]

        self.entropy = rng_ss.entropy

        self._neurons = neurons

        self._r = r
        self._m = m

        # interaction matrix and examples
        # examples get defined when k is changed
        # J is defined outside with set_interaction
        self._examples = None
        self._J = None


        # type of learning and whether diagonal is included
        self._supervised = supervised
        self._diagonal = diagonal
        self._t = t

        # initializes the patterns and examples (see k setter)
        self.k = k

        # the initial state and external field to be used in simulate
        # defined manually outside the constructor
        # self.external_field = None
        self.initial_state = None

    @property
    def neurons(self):
        return self._neurons

    @property
    def r(self):
        return self._r

    @property
    def m(self):
        return self._m

    @property
    def J(self):
        return self._J

    @property
    def patterns(self):
        return self._patterns

    @property
    def examples(self):
        return self._examples

    @property
    def k(self):
        return self._patterns.shape[0]

    @property
    def supervised(self):
        return self._supervised

    @property
    def t(self):
        return self._t

    @property
    def diagonal(self):
        return self._diagonal

    @supervised.setter
    def supervised(self, supervised):
        self._supervised = supervised
        self.set_interaction()

    @t.setter
    def t(self, t):
        self._t = t
        self.set_interaction()

    @diagonal.setter
    def diagonal(self, diagonal):
        self._diagonal = diagonal
        if diagonal:
            np.fill_diagonal(self._J, self.k / self.neurons)
        else:
            np.fill_diagonal(self._J, 0)

    # Changes the number of patterns by either generating new patterns and examples or removing them
    # Updates interaction matrix accordingly, only if it is already defined
    @k.setter
    def k(self, k):
        assert isinstance(k, int), 'Number of patterns must be an integer.'
        if self._examples is not None:
            if k > self.k:
                extra_patterns = self.gen_patterns(k-self.k)
                extra_examples = self.gen_examples(extra_patterns)
                self._patterns = np.concatenate((self._patterns, extra_patterns))
                self._examples = np.concatenate((self._examples, extra_examples), axis=1)
                if self._J is not None and self._t == 0:
                    self._J = self._J + self.interaction(extra_examples)
                elif self._J is not None and self._t > 0:
                    self.set_interaction()

            else:
                self._patterns = self._patterns[:k]
                self._examples = self._examples[:,:k]
                if self._J is not None:
                    self.set_interaction()

        else:
            self._patterns = self.gen_patterns(k)
            self._examples = self.gen_examples(self._patterns)

    # computes interaction matrix for a given set of examples
    # the reason this method and set_interaction are not the same method is for this one to be used for the extra patterns in add_patterns
    # that way we have an easy way to save time if iterating through alpha
    def interaction(self, examples = None):
        if examples is None:
            examples = self._examples
        c = self._m * self._neurons
        if self._supervised:
            av_examples = np.mean(self._examples, axis = 0)
            if self._t == 0:
                J = 1 / c * np.einsum('ui, uj -> ij', av_examples, av_examples)
            else:
                J = None
        else:
            if self._t == 0:
                J = 1 / c * np.einsum('aui, auj -> ij', examples, examples, optimize = True)
            else:
                J = None
        if not self._diagonal:
            np.fill_diagonal(J, 0)
        return J

    # constructor of the interaction matrix
    def set_interaction(self):
        self._J = self.interaction(self.examples)

    # generates patterns
    def gen_patterns(self, k):
        return self.noise_patterns.choice([-1, 1], (k, self._neurons))

    # generates examples
    def gen_examples(self, patterns):
        # the order of generation and transposition are to force the same examples to be generated regardless of
        # how k is changed

        patterns = self.patterns
        k = np.shape(patterns)[0]
        blurs = self.noise_examples.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2], size=(k, self.m, self._neurons))

        return np.transpose(blurs, [1, 0, 2]) * patterns


    def gen_samples(self, states, p, n=1):
        blurs = self.noise_examples.choice([-1, 1], p=[(1 - p) / 2, (1 + p) / 2], size=(n,) + np.shape(states))
        if n > 1:
            return blurs * states
        else:
            return blurs[0] * states

    def state(self, keyword, reduced = None):
        if keyword == 'arc' and (reduced == 'partial' or reduced is None):
            return self.patterns
        elif keyword == 'arc' and reduced == 'full':
            return self.patterns[0]
        elif keyword == 'ex' and reduced is None:
            return self.examples
        elif keyword == 'ex' and (reduced == 'full' or reduced == 'partial'):
            return self.examples[0]
        else:
            raise f'State {keyword} invalid.'

    # Method simulate runs the MonteCarlo simulation for 0 temperature
    # It does neurons flips per iteration.
    # Each of these neurons flips is one call of the function "dynamics_simple_noh" (defined above)
    # At each iteration it appends the new state a list
    # It loops until a maximum number of iterations is reached
    # Or until the standard deviation in the last av_counter magnetizations is below a certain threshold

    # INPUTS:
    # max_it is the maximum number of iterations
    # error is the threshold for difference between the last state and the one before
    # this is measured from 0 (same state) to 1 (symmetric states)

    # It returns the last state and the full history of errors
    def simulate_zero_T(self, max_it, error=0):

        state = self.initial_state

        errors = []
        idxs = tuple([np.empty(0) for _ in np.shape(state)])
        while len(errors) < max_it: # do the simulation

            old_state = state
            state = np.sign(state @ self.J)

            this_error = np.max((1 - np.mean(state * old_state, axis=-1)) / 2)
            diff = state - old_state
            prev_idxs = idxs
            idxs = np.nonzero(diff)
            errors.append(this_error)
            if this_error <= error or all([np.array_equal(x,y) for x, y in zip(idxs,prev_idxs)]):
                break

        return state, errors