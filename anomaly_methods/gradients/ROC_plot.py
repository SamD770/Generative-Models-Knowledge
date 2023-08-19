# from .unsupervised_sklearn import *
#
# from sklearn.metrics import roc_curve, auc, RocCurveDisplay
#
# import matplotlib.pyplot as plt
#
#
# def get_unsupervised_roc_curve(
#     id_test_data, id_train_data, ood_data_list, sklearn_model, fit_sample_proportion=0.8
# ):
#     def fit_roc_curve(data):
#         scores = sklearn_model.score_samples(data)
#
#         y_true = torch.cat([torch.ones(len(id_test)), torch.zeros(len(data))])
#
#         y_scores = torch.cat([torch.tensor(id_test_scores), torch.tensor(scores)])
#
#         fpr, tpr, _ = roc_curve(y_true, y_scores)
#
#         return fpr, tpr, auc(fpr, tpr)
#
#     id_fit, id_test, fit_samples = split_id_data(normed_id_norms, fit_sample_proportion)
#
#     sklearn_model.fit(id_fit)
#
#     id_test_scores = sklearn_model.score_samples(id_test)
#
#     ood_fpr = []
#     ood_tpr = []
#     ood_auc = []
#
#     for ood_data in ood_data_list:
#         fpr, tpr, roc_auc = fit_roc_curve(ood_data)
#
#         ood_fpr.append(fpr)
#         ood_tpr.append(tpr)
#         ood_auc.append(roc_auc)
#
#     return ood_fpr, ood_tpr, ood_auc
#
#
# if __name__ == "__main__":
#     id_dataset = "FashionMNIST"
#     dataset_names = ["FashionMNIST", "MNIST", "Omniglot"]
#
#     model_name = "FashionMNIST_stable"
#
#     for method in ["OneClassSVM"]:
#         for batch_size in [1]:
#             title = f"Norms ROC plot ({method}, {model_name}, Batch size {batch_size})"
#
#             print(f"fitting {title}")
#
#             if method == "Iso Forest":
#                 my_model = IsolationForest(n_estimators=10000)
#             elif method == "OneClassSVM":
#                 my_model = OneClassSVM(nu=0.001)
#             else:
#                 my_model = None
#
#             # Want: id_train_norms, id_test_norms, all_norms_list
#
#             id_norms, all_norms_list = get_norms(
#                 batch_size, model_name, id_dataset, dataset_names, printout=True
#             )
#
#             normed_id_norms, normed_all_norms_list = get_sklearn_norms(
#                 id_norms, all_norms_list
#             )
#
#             #
#             # id_test_data, id_train_data, ood_data_list \
#             #     = load_sklearn_norms(batch_size, model_name, id_dataset, dataset_names, printout=True)
#
#             fpr_list, tpr_list, auc_list = get_unsupervised_roc_curve(
#                 None, None, normed_all_norms_list, my_model
#             )
#
#             print(auc_list)
#
#             fig, ax = plt.subplots()
#
#             plt.title(title)
#
#             for fpr, tpr, roc_auc, dataset_name in zip(
#                 fpr_list, tpr_list, auc_list, dataset_names
#             ):
#                 print(dataset_name, roc_auc)
#                 display = RocCurveDisplay(
#                     fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=dataset_name
#                 )
#
#                 display.plot(ax=ax)
#
#             ax.run_multi_figures( "--", [0, 1], [0, 1],
#
#                                          # RocCurveDisplay(fpr=test_fpr, tpr=test_tpr, roc_auc=test_auc,
#                                          #                 estimator_name=f"{id_dataset}").plot(ax=ax)
#
#                                          plt.savefig(f"analysis/plots/ROC_plots/{title}.png")
