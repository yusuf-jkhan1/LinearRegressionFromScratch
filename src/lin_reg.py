import numpy as np
import math

class LinReg():

    n_iters = 1000
    learning_rate = 0.01

    def __init__(self):
        pass

    def _initialize_theta_vector(self, num_features):
        self.theta_vector = np.zeros((num_features,1))
        return None

    def _calculate_linear_combination(self, theta_vector, input_matrix):
        assert theta_vector.T.shape[1] == input_matrix.shape[0], "Dimensions don't match"
        self.lin_comb_vect = np.dot(self.theta_vector.T,self.X)
        return None

    def _calculate_basic_error(self, y, lin_comb_vect):
        self.error = y - lin_comb_vect
        return None

    def _compute_cost(self, y_vect, lin_comb_vect):
        errors = (y_vect - lin_comb_vect)
        squared_errors = errors**2
        mean_squared_error = np.mean(squared_errors)
        self.root_mean_squared_error = math.sqrt(mean_squared_error)
        return self.root_mean_squared_error

    def _normalize(self,X):
        mu = np.mean(self.X,1).reshape(self.num_features,1)
        sigma = np.std(self.X,1).reshape(self.num_features,1)
        self.X = (self.X - mu) / sigma
        return None

    def _add_bias_term(self,X):
        intercept_vector = np.ones((self.num_records,1)).T
        self.X = np.vstack((self.X,intercept_vector))
        return None   

    def _gradient_descent_step(self, X, y, learning_rate):
        
        self._calculate_linear_combination(self.theta_vector, self.X)

        self._calculate_basic_error(y, self.lin_comb_vect)       
       
        error_by_X = X @ self.error.T

        step_value = (learning_rate/self.num_records) * error_by_X

        self.theta_vector += step_value
 
        return None
    
    def _early_stop(self,cost_t0,cost_t1):
        percent_cost_change = (cost_t1 - cost_t0) / cost_t0 * 100
        print("Percent Change:",percent_cost_change)
        if abs(percent_cost_change) < self.percent_change_threshold:
            self.terminate = True
        else:
            self.terminate = False

    def fit(self,X,y):
        self.X = np.array(X).T
        self.y = np.array(y).T
        self.num_records = self.X.shape[1]
        self.num_features = self.X.shape[0]


        self._normalize(self.X)
        self._add_bias_term(self.X)
        self._initialize_theta_vector(self.num_features+1) # +1 for bias term
        self._calculate_linear_combination(self.theta_vector,self.X)

        for _ in range(self.n_iters):
            cost_t0 = self._compute_cost(self.y, self.lin_comb_vect)
            self._gradient_descent_step(self.X, self.y, self.learning_rate)
            cost_t1 = self._compute_cost(self.y, self.lin_comb_vect)

        self.tuned_theta_vector = self.theta_vector
        self.y_preds = self.theta_vector.T @ self.X
        self.score = 1 - (((self.y - self.y_preds)**2).sum() / ((self.y - self.y.mean())**2).sum())








