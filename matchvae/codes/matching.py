import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.covariance import MinCovDet, EmpiricalCovariance
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from scipy.stats import mannwhitneyu, kstest


class Matcher(object):
    def __init__(self, scores, t, propensity_key='propensity', att=True, atc=True):

        assert propensity_key in scores
        self.propensity_key = propensity_key

        self._scores = {}
        for key, score in scores.items():
            self._scores[key] = pd.DataFrame(score)

        self._t = pd.DataFrame(t)
        self.treated_indices = self._t.index[np.ravel(np.array(self._t == 1))]
        self.control_indices = self._t.index[np.ravel(np.array(self._t == 0))]

        self.att = att
        self.atc = atc

        self._distances_t_c = {}
        self._weights_att, self._weights_atc = self._init_weights()
        self._ites = {}

    def _init_weights(self):
        _weights_att = {}
        _weights_atc = {}
        for key in self._scores:
            _weights_att_key = pd.DataFrame(
                np.ones((len(self.treated_indices), len(self.control_indices))),
                index=self.treated_indices,
                columns=self.control_indices
            )
            _weights_att[key] = _weights_att_key if self.att else None
            _weights_atc[key] = _weights_att_key.T if self.atc else None
        return _weights_att, _weights_atc

    def _get_keys(self, on=None):
        if on is None:
            on = self._scores.keys()
        if isinstance(on, str):
            on = [on]
        if "balancing" in on:
            on = [el for el in on if 'balancing' not in el]
            for key in self._scores.keys():
                if 'balancing' in key:
                    on.append(key)
        return on

    def _clean_weights(self, key):
        self._weights_att[key] = self._weights_att[key].loc[
            self._weights_att[key].any(1), self._weights_att[key].any(0)] if self.att else None
        self._weights_atc[key] = self._weights_atc[key].loc[
            self._weights_atc[key].any(1), self._weights_atc[key].any(0)] if self.atc else None

    def _subsample(self, indices, on):
        for key in on:
            self._scores[key] = self._scores[key].loc[indices]
            treated_indices = self.treated_indices.intersection(indices)
            control_indices = self.control_indices.intersection(indices)

            self._distances_t_c[key] = self._distances_t_c[key].loc[treated_indices, control_indices]
            self._weights_att[key] = self._weights_att[key].loc[treated_indices, control_indices] if self.att else None
            self._weights_atc[key] = self._weights_atc[key].loc[control_indices, treated_indices] if self.atc else None
            self._clean_weights(key)

    def _update_weights(self, key, weights_att, weights_atc):

        if self.att:
            indices_att = self._weights_att[key].index.intersection(weights_att.index)
            if weights_att.shape[1] == 1:
                self._weights_att[key] = weights_att if self._weights_att[key].shape[1] > 1 \
                    else self._weights_att[key].loc[indices_att] * weights_att.loc[indices_att]
            else:
                columns_att = self._weights_att[key].columns.intersection(weights_att.columns)
                self._weights_att[key] = self._weights_att[key].loc[indices_att, columns_att] * weights_att.loc[
                    indices_att, columns_att]

        if self.atc:
            indices_atc = self._weights_atc[key].index.intersection(weights_atc.index)
            columns_atc = self._weights_atc[key].columns.intersection(weights_atc.columns)
            self._weights_atc[key] = self._weights_atc[key].loc[indices_atc, columns_atc] * weights_atc.loc[
                indices_atc, columns_atc]

        self._clean_weights(key)

    def trim_ps(self, eta=0.1, propensity_key=None, on=None):

        propensity_key = self.propensity_key if propensity_key is None else propensity_key
        ps = self._scores[propensity_key]
        indices = ps.index[np.ravel(np.array((ps >= eta) & (ps <= 1 - eta)))]

        on = self._get_keys(on)

        self._subsample(indices, on)

    def caliper(self, value, wrt=None, on=None):

        if wrt is None:
            wrt = self.propensity_key

        on = self._get_keys(on)

        _weights_att_caliper = (self._distances_t_c[wrt] <= value)
        _weights_atc_caliper = _weights_att_caliper.T

        for key in on:
            self._update_weights(key, _weights_att_caliper, _weights_atc_caliper)

    def _nearest_neighbor_matching_replacement_key(self, key, n_neighbors=3):

        weights_att = self._weights_att[key]
        weights_atc = self._weights_atc[key]

        if self.att:
            for treated_idx in weights_att.index:
                candidates_indices = weights_att.columns[weights_att.loc[treated_idx] > 0]
                if len(candidates_indices) > n_neighbors:
                    distances = self._distances_t_c[key].loc[treated_idx, candidates_indices]
                    distances = distances.sort_values()
                    weights_att.loc[treated_idx, distances.iloc[n_neighbors:].index] = 0

        if self.atc:
            for control_idx in weights_atc.index:
                candidates_indices = weights_atc.columns[weights_atc.loc[control_idx] > 0]
                if len(candidates_indices) > n_neighbors:
                    distances = self._distances_t_c[key].T.loc[control_idx, candidates_indices]
                    distances = distances.sort_values()
                    weights_atc.loc[control_idx, distances.iloc[n_neighbors:].index] = 0

        self._update_weights(key, weights_att, weights_atc)

    def _random_matching_replacement_key(self, key, n_neighbors=3):

        weights_att = self._weights_att[key]
        weights_atc = self._weights_atc[key]

        if self.att:
            for treated_idx in weights_att.index:
                candidates_indices = weights_att.columns[weights_att.loc[treated_idx] > 0]
                if len(candidates_indices) > n_neighbors:
                    weights_att.loc[treated_idx, np.random.permutation(candidates_indices)[n_neighbors:]] = 0

        if self.atc:
            for control_idx in weights_atc.index:
                candidates_indices = weights_atc.columns[weights_atc.loc[control_idx] > 0]
                if len(candidates_indices) > n_neighbors:
                    weights_atc.loc[control_idx, np.random.permutation(candidates_indices)[n_neighbors:]] = 0

        self._update_weights(key, weights_att, weights_atc)

    def nearest_neighbor_matching_replacement(self, n_neighbors=3, on=None):

        on = self._get_keys(on)

        for key in on:
            self._nearest_neighbor_matching_replacement_key(key, n_neighbors=n_neighbors)

    def random_matching_replacement(self, n_neighbors=3, on=None):

        on = self._get_keys(on)

        for key in on:
            self._random_matching_replacement_key(key, n_neighbors=n_neighbors)

    def ipw(self, on=None, eps=None):

        on = self._get_keys(on)
        print("IPW")
        for key in on:
            print(key)
            score = self._scores[key]
            assert score.shape[1] == 1

            if eps is not None:
                score[score < eps] = eps
                score[score > 1 - eps] = 1 - eps

            score.columns = self._t.columns

            weights_att = (self._t + (1 - self._t) * (score / (1 - score))).dropna('index') if self.att else None
            weights_atc = ((1 - self._t) + self._t * ((1 - score) / score)).dropna('index') if self.atc else None

            self._update_weights(key, weights_att, weights_atc)

    def cem_key(self, key, att=True):

        treated_indices = self.treated_indices if att else self.control_indices
        control_indices = self.control_indices if att else self.treated_indices

        score = np.array(self._scores[key])
        index = self._scores[key].index
        assignments = np.zeros(score.shape)

        for col in range(score.shape[1]):

            values = score[:, col]

            if len(np.unique(values)) > 2:  # continuous

                n_bins = np.ceil(np.log2(len(values)) + 1)
                diff = (np.max(values) - np.min(values)) / n_bins
                assignments_col = np.floor((values - np.min(values)) / diff)
                assignments_col[assignments_col == n_bins] = n_bins - 1

            else:  # binary

                assignments_col = values

            # discard variable if no intersection between treated and control
            assignments_col_indexed = pd.Series(assignments_col, index=index)
            if len(np.intersect1d(
                    assignments_col_indexed[assignments_col_indexed.index.intersection(treated_indices)],
                    assignments_col_indexed[assignments_col_indexed.index.intersection(control_indices)]
            )) == 0:
                assignments_col = np.zeros(assignments_col.shape)

            assignments[:, col] = assignments_col

        def get_string(vector):
            res = ''
            for el in vector.flatten():
                res += str(int(el)) + '-'
            return res[:-1]

        assignments_series = pd.Series(np.apply_along_axis(get_string, 1, assignments), index=index)
        print(assignments_series)
        weights = pd.Series(np.zeros(len(assignments_series)), index=index)

        n_treated_matched_total = 0
        n_control_matched_total = 0
        for assignment in pd.unique(assignments_series):
            indices_assignment_t = assignments_series.index[assignments_series == assignment].intersection(
                treated_indices)
            indices_assignment_c = assignments_series.index[assignments_series == assignment].intersection(
                control_indices)
            print(len(indices_assignment_t), len(indices_assignment_c), ' ---- ', assignment)
            if len(indices_assignment_t) > 0 and len(indices_assignment_c) > 0:
                weights[indices_assignment_t] = 1.
                weights[indices_assignment_c] = len(indices_assignment_t) / len(indices_assignment_c)
                n_treated_matched_total += len(indices_assignment_t)
                n_control_matched_total += len(indices_assignment_c)
        if (weights > 0).sum() > 0:
            weights.loc[weights.index[weights > 0].intersection(
                control_indices)] *= n_control_matched_total / n_treated_matched_total

        return weights

    def compute_distances(self, on=None, **kwargs):

        on = self._get_keys(on)
        print("Compute distances")

        for key in on:
            print(key)
            score = self._scores[key]
            score_t = score.loc[score.index.intersection(self.treated_indices)]
            score_c = score.loc[score.index.intersection(self.control_indices)]
            if 'metric' in kwargs and kwargs['metric'] == 'mahalanobis':
                mat = np.zeros((score_t.shape[0], score_c.shape[0]))

                if 'control_only' in kwargs and kwargs['control_only']:
                    score = score_c

                cov_mat_est = MinCovDet().fit(score) if 'covariance' in kwargs and kwargs['covariance'] == 'mincovdet' \
                    else EmpiricalCovariance().fit(score)

                for ind in range(score_t.shape[0]):
                    mat[ind] = np.sqrt(cov_mat_est.mahalanobis(np.array(score_t)[ind] - np.array(score_c)))
            else:
                mat = pairwise_distances(score_t, score_c, **kwargs)
            self._distances_t_c[key] = pd.DataFrame(mat, index=score_t.index, columns=score_c.index)

    def _get_treatment_effects_weighting(self, y, weights):
        y_weighted_all = y.loc[y.index.intersection(weights.index)] * weights.loc[weights.index.intersection(y.index)]
        y_1_mean = y_weighted_all.loc[y_weighted_all.index.intersection(self.treated_indices)].mean().item()
        y_0_mean = y_weighted_all.loc[y_weighted_all.index.intersection(self.control_indices)].mean().item()
        return y_1_mean - y_0_mean

    def _get_treatment_effects_key(self, y, key):

        weights_att = self._weights_att[key]
        weights_atc = self._weights_atc[key]

        ites_indices = []
        ites_values = []

        if self.att:
            if weights_att.shape[1] == 1:
                att = self._get_treatment_effects_weighting(y, weights_att)
            else:
                y_t_att = y.loc[y.index.intersection(weights_att.index)]
                y_c_att = y.loc[y.index.intersection(weights_att.columns)]
                weights_att = weights_att.loc[y_t_att.index, y_c_att.index]
                att = 0.
                for treated_idx in weights_att.index:
                    y_1 = y_t_att.loc[treated_idx].item()
                    weights = np.ravel(np.array(weights_att.loc[treated_idx]))
                    y_0_s = np.ravel(np.array(y_c_att))
                    y_0 = np.sum(y_0_s * weights) / weights.sum()

                    att += y_1 - y_0

                    ites_indices.append(treated_idx)
                    ites_values.append(y_1 - y_0)

                att /= len(weights_att.index)
        else:
            att = None

        if self.atc:
            if weights_atc.shape[1] == 1:
                atc = self._get_treatment_effects_weighting(y, weights_atc)
            else:
                y_c_atc = y.loc[y.index.intersection(weights_atc.index)]
                y_t_atc = y.loc[y.index.intersection(weights_atc.columns)]
                weights_atc = weights_atc.loc[y_c_atc.index, y_t_atc.index]
                atc = 0.
                for control_idx in weights_atc.index:
                    y_0 = y_c_atc.loc[control_idx].item()
                    weights = np.ravel(np.array(weights_atc.loc[control_idx]))
                    y_1_s = np.ravel(np.array(y_t_atc))
                    y_1 = np.sum(y_1_s * weights) / weights.sum()
                    atc += y_1 - y_0
                    ites_indices.append(control_idx)
                    ites_values.append(y_1 - y_0)
                atc /= len(weights_atc.index)
        else:
            atc = None

        ate = None
        if self.att and self.atc:
            ate = att * len(weights_att.index) + atc * len(weights_atc.index)
            ate /= len(weights_att.index) + len(weights_atc.index)

        self._ites[key] = pd.DataFrame({
            'ites': ites_values
        }, index=ites_indices)

        return att, atc, ate

    def _get_pehes(self, key, ites_gt):

        ites_gt = pd.DataFrame(ites_gt)
        ites_pred = self._ites[key]
        ites_gt = ites_gt.loc[ites_gt.index.intersection(ites_pred.index)]
        ites_pred = ites_pred.loc[ites_pred.index.intersection(ites_gt.index)]

        if len(ites_pred) == 0:
            return None, None, None

        ites_gt_t = ites_gt.loc[ites_gt.index.intersection(self.treated_indices)]
        ites_gt_c = ites_gt.loc[ites_gt.index.intersection(self.control_indices)]
        ites_pred_t = ites_pred.loc[ites_pred.index.intersection(self.treated_indices)]
        ites_pred_c = ites_pred.loc[ites_pred.index.intersection(self.control_indices)]

        pehe_all = np.sqrt(np.mean(np.power(np.array(ites_gt - ites_pred), 2)))
        pehe_treated = np.sqrt(np.mean(np.power(np.array(ites_gt_t - ites_pred_t), 2))) if len(
            ites_pred_t) > 0 else None
        pehe_control = np.sqrt(np.mean(np.power(np.array(ites_gt_c - ites_pred_c), 2))) if len(
            ites_pred_c) > 0 else None

        return pehe_all, pehe_treated, pehe_control

    def get_treatment_effects(self, y, evaluate=True, ites=None, on=None):

        print("Get treatment effects")

        y = pd.DataFrame(y)

        atts = []
        atcs = []
        ates = []

        on = self._get_keys(on)

        for key in on:
            print(key)
            att, atc, ate = self._get_treatment_effects_key(y, key)
            atts.append(att)
            atcs.append(atc)
            ates.append(ate)

        results = pd.DataFrame(dict(att=atts, atc=atcs, ate=ates), index=on)

        if evaluate and ites is not None:
            ites = pd.DataFrame({
                'ites': np.ravel(ites)
            })
            ites_t = ites.loc[ites.index.intersection(self.treated_indices)]
            ites_c = ites.loc[ites.index.intersection(self.control_indices)]

            results = results.append(pd.DataFrame(dict(
                att=np.mean(np.array(ites_t)) if self.att else None,
                atc=np.mean(np.array(ites_c)) if self.atc else None,
                ate=np.mean(np.array(ites)),
            ), index=['ground truth']))
            results['error_att'] = np.abs(results['att'] - np.mean(np.array(ites_t))) if self.att else None
            results['error_atc'] = np.abs(results['atc'] - np.mean(np.array(ites_c))) if self.atc else None
            results['error_ate'] = np.abs(results['ate'] - np.mean(np.array(ites)))
            results['pehe_all'] = np.zeros((len(results.index)))
            results['pehe_treated'] = np.zeros((len(results.index)))
            results['pehe_control'] = np.zeros((len(results.index)))
            for key in on:
                pehe_all, pehe_treated, pehe_control = self._get_pehes(key, ites)
                results.loc[key, 'pehe_all'] = pehe_all
                results.loc[key, 'pehe_treated'] = pehe_treated
                results.loc[key, 'pehe_control'] = pehe_control

        return results

    def make_matched_sample(self, key, att=True):
        treated_indices = []
        control_indices = []

        weights_att = self._weights_att[key]
        weights_atc = self._weights_atc[key]

        if att:
            if weights_att.shape[1] == 1:
                treated_indices = weights_att.index.intersection(self.treated_indices)
                control_indices = weights_att.index.intersection(self.control_indices)
            else:
                treated_indices = weights_att.index
                control_indices = weights_att.columns

        else:
            if weights_att.shape[1] == 1:
                treated_indices = weights_atc.index.intersection(self.treated_indices)
                control_indices = weights_atc.index.intersection(self.control_indices)
            for control_idx in weights_atc.index:
                control_indices.append(control_idx)
                for treated_idx in weights_atc.columns:
                    if weights_atc.loc[control_idx, treated_idx].item() > 0:
                        if treated_idx not in treated_indices:
                            treated_indices.append(treated_idx)

        return treated_indices, control_indices

    def get_balance_metrics(self, x, on=None, add_nothing=False):

        x = pd.DataFrame(x)

        on = self._get_keys(on)

        distances = pairwise_distances(np.array(x))
        sigma = np.median(distances)
        distances = pd.DataFrame(distances, index=x.index, columns=x.index)

        linear_mmds_att = []
        # standardised_diffs_norms_att = []
        linear_mmds_atc = []
        # standardised_diffs_norms_atc = []

        if add_nothing:
            key_original = 'no matching'
            self._weights_att[key_original] = pd.DataFrame(
                np.ones((len(self.treated_indices), len(self.control_indices))),
                index=self.treated_indices,
                columns=self.control_indices
            )
            self._weights_atc[key_original] = self._weights_att[key_original].T

            on = list(on) + [key_original]
        print("Get balance metrics")
        for key in on:
            print(key)
            if self.att:
                # ATT

                treated_indices, control_indices = self.make_matched_sample(key, att=True)
                x_t = x.loc[x.index.intersection(treated_indices)]
                x_c = x.loc[x.index.intersection(control_indices)]

                linear_mmds_att.append(np.sum(np.power(np.array(x_t).mean(axis=0) - np.array(x_c).mean(axis=0), 2)))
                # standardised_diffs_norms_att.append(np.sum(
                #    2*np.power(np.array(x_t).mean(axis=0) - np.array(x_c).mean(axis=0),2)\
                #    / (np.array(x_t).var(axis=0) + np.array(x_c).var(axis=0))
                # ))
            else:
                linear_mmds_att.append(None)

            if self.atc:
                # ATC
                treated_indices, control_indices = self.make_matched_sample(key, att=False)
                x_t = x.loc[x.index.intersection(treated_indices)]
                x_c = x.loc[x.index.intersection(control_indices)]
                linear_mmds_atc.append(np.sum(np.power(np.array(x_t).mean(axis=0) - np.array(x_c).mean(axis=0), 2)))
                # standardised_diffs_norms_atc.append(np.sum(
                #    2*np.power(np.array(x_t).mean(axis=0) - np.array(x_c).mean(axis=0),2)\
                #    / (np.array(x_t).var(axis=0) + np.array(x_c).var(axis=0))
                # ))
            else:
                linear_mmds_atc.append(None)

        results = pd.DataFrame(dict(
            linear_mmd_att=linear_mmds_att,
            linear_mmd_atc=linear_mmds_atc,
        ), index=on)

        if add_nothing:
            del self._weights_att[key_original]
            del self._weights_atc[key_original]

        return results

    def apply_programme(self, methods_list):
        results = []
        for el in methods_list:
            method, kwargs = el
            results.append(getattr(self, method)(**kwargs))
        return results


PROGRAMME_DEFAULT = [
            ('compute_distances', dict(on='propensity')),
            ('compute_distances', dict(metric='euclidean', on=['z','balancing'])),
            #('trim_ps', dict(eta=0.1, on=['z','balancing','propensity'])),
            #('caliper', dict(value=0.1, on='propensity')),
            ('nearest_neighbor_matching_replacement', dict(n_neighbors=150, on=['z','balancing','propensity']))
]

METRICS_DEFAULT = ['att','error_att','pehe_treated','linear_mmd_att','fraction_of_predicted_treated','accuracy','cross-entropy']

aggreg_funcs = {
    'median' : np.median,
    'min' : np.min,
    'max' : np.max
}


def matching_metrics(model, data, data_to_add=None, t_to_add=0, programme=None, metrics=None, aggreg_matching='median'):
    x, y, t, mu0, mu1, cate_true = data[:]
    if data_to_add is not None:
        x_add, y_add, t_add, mu0_add, mu1_add, cate_true_add = data_to_add[:]
        mask = (t_add == t_to_add).ravel()
        x_add = x_add[mask]
        y_add = y_add[mask]
        mu0_add = mu0_add[mask]
        mu1_add = mu1_add[mask]
        cate_true_add = cate_true_add[mask]
        t_add = t_add[mask]
        x = torch.vstack((x, x_add))
        y = torch.vstack((y, y_add))
        t = torch.vstack((t, t_add))
        mu0 = torch.vstack((mu0, mu0_add))
        mu1 = torch.vstack((mu1, mu1_add))
        cate_true = torch.vstack((cate_true, cate_true_add))

    y_numpy = y.cpu().detach().numpy()
    t_numpy = t.cpu().detach().numpy()
    cate_numpy = cate_true.cpu().detach().numpy()

    if metrics is None:
        metrics = METRICS_DEFAULT

    print("Generating matchings")
    model.eval()
    _, _, q, _ = model(x, t)
    z = q.mean
    scores = model.decoder_t.get_scores(z)
    scores['z'] = z.cpu().detach().numpy()

    if programme is None:
        programme = [el for el in PROGRAMME_DEFAULT]
    programme += [
        ('get_treatment_effects', dict(y=y_numpy, evaluate=True, ites=cate_numpy)),
        ('get_balance_metrics', dict(x=x.cpu().detach().numpy(), add_nothing=True)),
    ]

    print([el[0] for el in programme])
    ptz_0 = scores['propensity'][t_numpy == 0].ravel()
    ptz_1 = scores['propensity'][t_numpy == 1].ravel()
    m = Matcher(scores, t.cpu().detach().numpy(), att=True, atc=False)
    results = m.apply_programme(programme)
    decoder_t_results = {
        'fraction_of_predicted_treated': (scores['propensity'] > 0.5).mean(),
        'accuracy': (t_numpy == (scores['propensity'] > 0.5)).mean(),
        'cross-entropy': -(
                    t_numpy * np.log(scores['propensity']) + (1 - t_numpy) * np.log(1 - scores['propensity'])).mean(),
        'propensity_mean_difference': np.abs(np.mean(ptz_1) - np.mean(ptz_0)),
        'propensity_non_extremes': np.mean((scores['propensity'] > 0.1) * (scores['propensity'] < 0.9)),
        'propensity_mw_normalized': mannwhitneyu(ptz_0, ptz_1, use_continuity=True, alternative='less')[0] / (ptz_0.size * ptz_1.size),
        'propensity_ks': kstest(ptz_0, ptz_1)[0],
    }

    balance_df = results[-1]
    te_df = results[-2]

    print("Generating results")

    matching_results = {}
    for df in [te_df, balance_df]:
        matching_results.update(
            {
                method + '_' + metric: df.loc[method, metric] \
                for metric in df.columns if metric in metrics \
                for method in df.index
            }
        )
        matching_results.update(
            {
                aggreg_matching + '_' + metric: aggreg_funcs[aggreg_matching](np.array([df.loc[method, metric] \
                                                                                        for method in df.index if
                                                                                        method not in ['x',
                                                                                                       'no matching',
                                                                                                       'ground truth']])) \
                for metric in df.columns if metric in metrics \
 \
                }
        )

    return matching_results, decoder_t_results, dict(**scores)


