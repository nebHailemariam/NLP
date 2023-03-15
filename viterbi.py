class TNode:
    def __init__(self, v_path, v_prob) -> None:
        self.v_path = v_path
        self.v_prob = v_prob


class Viterbi:
    def __init__(self) -> None:
        self.states = ["#", "NN", "VB"]
        self.observations = ["I", "write", "a letter"]

        self.start_probabilities = [0.3, 0.4, 0.3]

        self.transition_probabilities = [
            [0.2, 0.2, 0.6],
            [0.4, 0.1, 0.5],
            [0.1, 0.8, 0.1],
        ]
        self.emission_probabilities = [
            [0.01, 0.02, 0.02],
            [0.8, 0.01, 0.5],
            [0.19, 0.97, 0.48],
        ]

    def forward(
        self,
        observations,
        states,
        start_probabilities,
        transition_probabilities,
        emission_probabilities,
    ):
        T = [None for s in states]

        for state in range(len(states)):
            path = [state]
            T[state] = TNode(
                path, start_probabilities[state] * emission_probabilities[state][0]
            )

        for output in range(1, len(observations)):
            U = [None for s in states]

            for next_state in range(len(states)):
                argmax = []
                valmax = 0

                for state in range(len(states)):
                    v_path = T[state].v_path[:]
                    v_prob = T[state].v_prob

                    p = (
                        emission_probabilities[next_state][output]
                        * transition_probabilities[state][next_state]
                    )
                    v_prob *= p

                    if v_prob > valmax:
                        if len(v_path) == len(states):
                            argmax = v_path
                        else:
                            v_path.append(next_state)
                            argmax = v_path
                        valmax = v_prob
                U[next_state] = TNode(argmax, valmax)
            T = U

        argmax = []
        valmax = 0

        for state in range(len(states)):
            v_path = T[state].v_path
            v_prob = T[state].v_prob

            if v_prob > valmax:
                argmax = v_path
                valmax = v_prob

        for i in range(len(argmax)):
            print(states[argmax[i]])

    def test(self):
        self.forward(
            self.observations,
            self.states,
            self.start_probabilities,
            self.transition_probabilities,
            self.emission_probabilities,
        )


viterbi = Viterbi()
viterbi.test()
