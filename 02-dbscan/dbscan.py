import numpy as np

class DBC():

    def __init__(self, dataset, min_pts, epsilon):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon


    def _get_neighborhood(self, Pn):
        neighborhood = []
        for P in range(len(self.dataset)):
            if np.linalg.norm(self.dataset[P] - self.dataset[Pn]) <= self.epsilon:
                neighborhood.append(P)
        return neighborhood

    
    def _assign_cluster(self, P, Neighborhood, assignment, labels):
        labels[P] = assignment
        while Neighborhood:
            Pn = Neighborhood.pop()
            if labels[Pn] == ~1:
                labels[Pn] = assignment
            elif labels[Pn] == 0:
                labels[Pn] = assignment
                new_neighborhood = self._get_neighborhood(Pn)

                if len(new_neighborhood) >= self.min_pts:
                    Neighborhood += new_neighborhood
            
        return labels


    def dbscan(self):
        labels = [0 for i in range(len(self.dataset))]
        assignment = 0
        for P in range(len(self.dataset)):
            if not (labels[P] == 0):
                # 
                continue

            PNeighborhood = self._get_neighborhood(P)
            if len(PNeighborhood) < self.min_pts:
                labels[P] = ~1
            else:
                assignment += 1
                labels = self._assign_cluster(P, PNeighborhood, assignment, labels)

        return labels
