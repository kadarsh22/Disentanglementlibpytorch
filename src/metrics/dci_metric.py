from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from utils import *



class DCIMetric(object):
    """ Impem
entation of the metric in:
        A FRAMEWORK FOR THE QUANTITATIVE EVALUATION OF DISENTANGLED
        REPRESENTATIONS
        Part of the code is adapted from:
        https://github.com/cianeastwood/qedr
    """
    def __init__(self, metric_data, regressor="Lasso", *args, **kwargs):
        super(DCIMetric, self).__init__(*args, **kwargs)
        self.data = metric_data["img_with_latent"]["img"]
        self.latents = metric_data["img_with_latent"]["latent"]

        self._regressor = regressor
        if regressor == "Lasso":
            self.regressor_class = Lasso
            self.alpha = 0.02
            # constant alpha for all models and targets
            self.params = {"alpha": self.alpha}
            # weights
            self.importances_attr = "coef_"
        elif regressor == "LassoCV":
            self.regressor_class = LassoCV
            # constant alpha for all models and targets
            self.params = {}
            # weights
            self.importances_attr = "coef_"
        elif regressor == "RandomForest":
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search
            max_depths = [4, 5, 2, 5, 5]
            # Create the parameter grid based on the results of random search
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestIBGAN":
            # The parameters that IBGAN paper uses
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search
            max_depths = [4, 2, 4, 2, 2]
            # Create the parameter grid based on the results of random search
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestCV":
            self.regressor_class = GridSearchCV
            # Create the parameter grid based on the results of random search
            param_grid = {"max_depth": [i for i in range(2, 16)]}
            self.params = {
                "estimator": RandomForestRegressor(),
                "param_grid": param_grid,
                "cv": 3,
                "n_jobs": -1,
                "verbose": 0
            }
            self.importances_attr = "feature_importances_"
        elif "RandomForestEnum" in regressor:
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search
            self.params = {
                "max_depth": int(regressor[len("RandomForestEnum"):]),
                "oob_score": True
            }
            self.importances_attr = "feature_importances_"
        else:
            raise NotImplementedError()

        self.TINY = 1e-12

    def normalize(self, X):
        mean = np.mean(X, 0) # training set
        stddev = np.std(X, 0) # training set
        #print('mean', mean)
        #print('std', stddev)
        return (X - mean) / stddev

    def norm_entropy(self, p):
        '''p: probabilities '''
        n = p.shape[0]
        return - p.dot(np.log(p + self.TINY) / np.log(n + self.TINY))

    def entropic_scores(self, r):
        '''r: relative importances '''
        r = np.abs(r)
        ps = r / np.sum(r, axis=0) # 'probabilities'
        hs = [1 - self.norm_entropy(p) for p in ps.T]
        return hs

    def evaluate(self, model):
        codes,_ = model.encoder(torch.FloatTensor(self.data))
        codes = codes.detach().cpu().numpy()
        latents = self.latents
        codes = self.normalize(codes)
        latents = self.normalize(latents)
        R = []

        for j in range(self.latents.shape[-1]):
            if isinstance(self.params, dict):
              regressor = self.regressor_class(**self.params)
            elif isinstance(self.params, list):
              regressor = self.regressor_class(**self.params[j])
            regressor.fit(codes, latents[:, j])

            # extract relative importance of each code variable in
            # predicting the latent z_j
            if self._regressor == "RandomForestCV":
                best_rf = regressor.best_estimator_
                r = getattr(best_rf, self.importances_attr)[:, None]
            else:
                r = getattr(regressor, self.importances_attr)[:, None]

            R.append(np.abs(r))

        R = np.hstack(R) #columnwise, predictions of each z

        # disentanglement
        disent_scores = self.entropic_scores(R.T)
        # relative importance of each code variable
        c_rel_importance = np.sum(R, 1) / np.sum(R)
        disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)

        # completeness
        complete_scores = self.entropic_scores(R)
        complete_avg = np.mean(complete_scores)

        return {
            "DCI_{}_disent_metric_detail".format(self._regressor): \
                disent_scores,
            "DCI_{}_disent_metric".format(self._regressor): disent_w_avg,
            "DCI_{}_complete_metric_detail".format(self._regressor): \
                complete_scores,
            "DCI_{}_complete_metric".format(self._regressor): complete_avg,
            "DCI_{}_metric_detail".format(self._regressor): R
            }


