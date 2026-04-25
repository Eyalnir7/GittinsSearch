import numpy as np
import enum

class BanditType(enum.Enum):
    LOOP=1
    LINE=2

class MarkovChain:
    beta = 0.999
    def __init__(self, done_transitions: np.array, done_times: np.array, fail_transitions: np.array, fail_times: np.array, type: BanditType):
        # the arrays are interpreted as:
        # state done_times[i] has a transition to DONE with probability done_transitions[i]
        self.done_transitions = done_transitions
        self.done_times = done_times
        self.fail_transitions = fail_transitions
        self.fail_times = fail_times
        self.type = type
        if 0 not in self.done_times:
            self.done_times = np.concatenate(([0], self.done_times))
            self.done_transitions = np.concatenate(([0.0], self.done_transitions))
        if 0 not in self.fail_times:
            self.fail_times = np.concatenate(([0], self.fail_times))
            self.fail_transitions = np.concatenate(([0.0], self.fail_transitions))
        if self.fail_times[-1] == self.done_times[-1]:
            self.fail_transitions[-1] = 1- self.done_transitions[-1]
        elif self.fail_times[-1] < self.done_times[-1]:
            self.fail_times = np.concatenate((self.fail_times, [self.done_times[-1]]))
            self.fail_transitions = np.concatenate((self.fail_transitions, [1.0 - self.done_transitions[-1]]))
        else:
            self.fail_transitions[-1] = 1.0
        self.all_times = np.sort(np.unique(np.concatenate((self.done_times, self.fail_times))))

        # assert the the pair of arrays have the same length
        if len(self.done_transitions) != len(self.done_times):
            print(f"Length mismatch: done_transitions={len(self.done_transitions)}, done_times={len(self.done_times)}")
            print(f"done_transitions: {self.done_transitions}")
            print(f"done_times: {self.done_times}")
        assert len(self.done_transitions) == len(self.done_times)
        
        if len(self.fail_transitions) != len(self.fail_times):
            print(f"Length mismatch: fail_transitions={len(self.fail_transitions)}, fail_times={len(self.fail_times)}")
        assert len(self.fail_transitions) == len(self.fail_times)

        # assert that the transition probabilites are less than 1
        assert np.all((self.done_transitions >= 0) & (self.done_transitions <= 1))
        assert np.all((self.fail_transitions >= 0) & (self.fail_transitions <= 1))

    def step(self):
        done_index = 0
        fail_index = 0
        completed = False
        completed_state = 0
        prev_t = 0
        for i in range(len(self.all_times)):
            t = self.all_times[i]
            while prev_t < t:
                yield 0
                prev_t += 1
            p_done = 0.0
            p_fail = 0.0
            if done_index < len(self.done_times) and t == self.done_times[done_index]:
                p_done = self.done_transitions[done_index]
                done_index += 1
            if fail_index < len(self.fail_times) and t == self.fail_times[fail_index]:
                p_fail = self.fail_transitions[fail_index]
                fail_index += 1
            # sample according to the probabilities
            r = np.random.random()
            if r < p_done:
                completed = True
                completed_state = 1
                yield 1
            if r < p_done + p_fail:
                completed = True
                completed_state = -1
                yield -1
            else:
                yield 0
            prev_t = t + 1
        if completed:
            yield completed_state

    # I want to be able to do MarkovChain[i:j] to get a sub-chain
    def get_subchain(self, start, end):
        fail_indexes = (self.fail_times >= start) & (self.fail_times <= end)
        done_indexes = (self.done_times >= start) & (self.done_times <= end)
        return MarkovChain(self.done_transitions[done_indexes], self.done_times[done_indexes], self.fail_transitions[fail_indexes], self.fail_times[fail_indexes], self.type)
    
    def get_gittins_numerator(self, next_layer_numerator=1/(1-beta)):
        # return the value a_0 = E[beta^T/(1-beta)|s0=0] and T is the hitting time in the DONE state. If the chain is not the last layer, next_layer_numerator is the numerator of the next layer.
        # in the following we solve the linear system a_t1 = beta**(t2-t1)*p_next*a_t2 + beta*p_done*next_layer_numerator + beta*p_fail*a_0 where a_s is E[beta^T/(1-beta)|s0=s]
        if self.type == BanditType.LOOP:
            size = len(self.all_times)
            matrix = np.zeros((size, size))
            rhs = np.zeros(size)
            done_index = 0
            fail_index = 0
            i = 0
            while i < len(self.all_times):
                t1 = self.all_times[i]
                if i < len(self.all_times) - 1:
                    t2 = self.all_times[i+1]
                p_done = 0.0
                p_fail = 0.0
                if done_index < len(self.done_times) and t1 == self.done_times[done_index]:
                    p_done = self.done_transitions[done_index]
                    done_index += 1
                if fail_index < len(self.fail_times) and t1 == self.fail_times[fail_index]:
                    p_fail = self.fail_transitions[fail_index]
                    fail_index += 1
                next_prob = 1.0 - p_done - p_fail
                matrix[i, 0] = MarkovChain.beta*p_fail
                rhs[i] = MarkovChain.beta*p_done*next_layer_numerator
                if i < len(self.all_times) - 1:
                    matrix[i, i+1] = MarkovChain.beta**(t2 - t1)*next_prob
                i += 1
            # solve the linear system
            a = np.linalg.solve(np.eye(size) - matrix, rhs)
            return a[0] # a[i] is a_i
        
        if self.type == BanditType.LINE:
            size = len(self.all_times)
            matrix = np.zeros((size, size))
            rhs = np.zeros(size)
            done_index = 0
            fail_index = 0
            i = 0
            while i < len(self.all_times):
                t1 = self.all_times[i]
                if i < len(self.all_times) - 1:
                    t2 = self.all_times[i+1]
                p_done = 0.0
                p_fail = 0.0
                if done_index < len(self.done_times) and t1 == self.done_times[done_index]:
                    p_done = self.done_transitions[done_index]
                    done_index += 1
                if fail_index < len(self.fail_times) and t1 == self.fail_times[fail_index]:
                    p_fail = self.fail_transitions[fail_index]
                    fail_index += 1
                next_prob = 1.0 - p_done - p_fail
                rhs[i] = MarkovChain.beta*p_done*next_layer_numerator
                if i < len(self.all_times) - 1:
                    matrix[i, i+1] = MarkovChain.beta**(t2 - t1)*next_prob
                i += 1
            # solve the linear system
            a = np.linalg.solve(np.eye(size) - matrix, rhs)
            return a[0] # a[i] is a_i
        
    def get_gittins_denominator_aux(self, next_layer_aux=0.0):
        # The denominator is 1/(1-beta) - E[beta**T/(1-beta)] where T is the hitting time in a FAIL state. We compute E[beta**T/(1-beta)] via solving a linear system similar to the numerator case. In this function we return E[beta**T/(1-beta)] and the caller can compute the denominator.
        if self.type == BanditType.LOOP:
            return next_layer_aux
        if self.type == BanditType.LINE:
            size = len(self.all_times)
            matrix = np.zeros((size, size))
            rhs = np.zeros(size)
            done_index = 0
            fail_index = 0
            i = 0
            while i < len(self.all_times):
                t1 = self.all_times[i]
                if i < len(self.all_times) - 1:
                    t2 = self.all_times[i+1]
                p_done = 0.0
                p_fail = 0.0
                if done_index < len(self.done_times) and t1 == self.done_times[done_index]:
                    p_done = self.done_transitions[done_index]
                    done_index += 1
                if fail_index < len(self.fail_times) and t1 == self.fail_times[fail_index]:
                    p_fail = self.fail_transitions[fail_index]
                    fail_index += 1
                next_prob = 1.0 - p_done - p_fail
                rhs[i] = MarkovChain.beta*p_done*next_layer_aux + MarkovChain.beta*p_fail*1/(1 - MarkovChain.beta)
                if i < len(self.all_times) - 1:
                    matrix[i, i+1] = MarkovChain.beta**(t2 - t1)*next_prob
                i += 1
            # solve the linear system
            c = np.linalg.solve(np.eye(size) - matrix, rhs)
            return c[0] # c[i] is c_i
    
    def get_stopping_time_and_gittins_parts(self, state, next_layer_numerator=1/(1-beta), next_layer_aux=0.0):
        start_index_chain = self.get_subchain(state, np.inf)
        optimal_stopping_time = None
        optimal_numerator = None
        optimal_denominator_aux = None
        optimal_gi = -np.inf
        for i in range(len(start_index_chain.all_times)):
            end_index_chain = start_index_chain.get_subchain(-np.inf, start_index_chain.all_times[i])
            numerator= end_index_chain.get_gittins_numerator(next_layer_numerator)
            denominator_aux = end_index_chain.get_gittins_denominator_aux(next_layer_aux)
            denominator = 1/(1-MarkovChain.beta)-denominator_aux
            gi = numerator/denominator
            if gi > optimal_gi:
                optimal_gi = gi
                optimal_stopping_time = start_index_chain.all_times[i]
                optimal_numerator = numerator
                optimal_denominator_aux = denominator_aux
        return optimal_stopping_time, optimal_numerator, optimal_denominator_aux


class BanditProcess:
    beta = 0.999
    def __init__(self, markov_chains: list[MarkovChain], bandit_types: list[BanditType]):
        self.markov_chains = markov_chains
        self.bandit_types = bandit_types
        self.state = 0

    def get_gittins_index(self):
        next_layer_numerator = 1/(1-BanditProcess.beta)
        next_layer_aux = 0.0
        for i in reversed(range(len(self.markov_chains))):
            chain = self.markov_chains[i]
            chain.type = self.bandit_types[i]
            chain_state = self.state if i == 0 else 0
            stopping_time, numerator, denominator_aux = chain.get_stopping_time_and_gittins_parts(chain_state, next_layer_numerator, next_layer_aux)
            if chain.type == BanditType.LOOP:
                chain = chain.get_subchain(-np.inf, stopping_time)
                numerator = chain.get_gittins_numerator(next_layer_numerator)
                denominator_aux = chain.get_gittins_denominator_aux(next_layer_aux)
            next_layer_numerator = numerator
            next_layer_aux = denominator_aux
        gittins_index = numerator/(1/(1-BanditProcess.beta)-denominator_aux)
        return gittins_index, stopping_time
    
if __name__=="__main__":
    # Example usage
    done_transitions = np.array([0.5])
    done_times = np.array([1])
    fail_transitions = np.array([0.5])
    fail_times = np.array([1])
    markov_chain = MarkovChain(done_transitions, done_times, fail_transitions, fail_times, BanditType.LINE)
    bandit_process = BanditProcess([markov_chain], [BanditType.LINE])
    gi, stopping_time = bandit_process.get_gittins_index()
    print(f"Gittins index: {gi}, Stopping time: {stopping_time}")
    print((BanditProcess.beta**2)/(2*(1+BanditProcess.beta)*(1-BanditProcess.beta)+BanditProcess.beta**2))

    done_transitions = np.array([0.6,0.6])
    done_times = np.array([0,2])
    fail_transitions = np.array([0.4])
    fail_times = np.array([1])
    markov_chain = MarkovChain(done_transitions, done_times, fail_transitions, fail_times, BanditType.LOOP)
    bandit_process = BanditProcess([markov_chain], [BanditType.LOOP])
    gi, stopping_time = bandit_process.get_gittins_index()
    print(f"Gittins index: {gi}, Stopping time: {stopping_time}")
    b = BanditProcess.beta
    V1 = (0.6*b + 0.4*0.6**2*b**3)/(1- 0.4**2*b**2 - 0.6*0.4**2*b**3) # this is if the stopping time is 2 but it is zero
    print(V1)
    print(0.6*b/(1-0.4*b))

    done_transitions = np.array([1])
    done_times = np.array([151])
    fail_transitions = np.array([])
    fail_times = np.array([])
    markov_chain1 = MarkovChain(done_transitions, done_times, fail_transitions, fail_times, BanditType.LINE)
    bandit_process = BanditProcess([markov_chain1], [BanditType.LINE])
    gi, stopping_time = bandit_process.get_gittins_index()
    print(f"Gittins index: {gi}, Stopping time: {stopping_time}")
    print(BanditProcess.beta**151/(1-BanditProcess.beta**151))

    done_transitions = np.array([0.5])
    done_times = np.array([101])
    fail_transitions = np.array([0.5])
    fail_times = np.array([101])
    markov_chain2 = MarkovChain(done_transitions, done_times, fail_transitions, fail_times, BanditType.LINE)
    bandit_process = BanditProcess([markov_chain2], [BanditType.LINE])
    gi, stopping_time = bandit_process.get_gittins_index()
    print(f"Gittins index: {gi}, Stopping time: {stopping_time}")

    done_transitions = np.array([0.5, 1.0])
    done_times = np.array([101, 201])
    fail_transitions = np.array([])
    fail_times = np.array([])
    markov_chain2 = MarkovChain(done_transitions, done_times, fail_transitions, fail_times, BanditType.LINE)
    bandit_process = BanditProcess([markov_chain2], [BanditType.LINE])
    gi, stopping_time = bandit_process.get_gittins_index()
    print(f"Gittins index: {gi}, Stopping time: {stopping_time}")

    done_transitions = np.array([0.5, 1.0])
    done_times = np.array([1, 100])
