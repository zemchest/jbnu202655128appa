# common
data_label_csv = 'Data Label.csv'
data_series_csv = 'Data Series.csv'
data_case_info_txt = 'Data Case Info.txt'
data_length_info_txt = 'Data Length Info.txt'
min = 'Min'
max = 'Max'
median = 'Median'
a_mean = 'Arithmetic Mean'
g_mean = 'Geometric Mean'
h_mean = 'Harmonic Mean'
std = 'Standard Deviation'
skew = 'Skewness'
kurt = 'Kurtosis'
common_plot_fig_size = (12, 9)
common_plot_title = True
common_plot_title_font_size = 30
common_plot_label_font_size = 30
common_plot_tick_font_size = 20
common_plot_border = True
# 1 merge Z50
case = 'Case'
torque_nm = 'Torque (Nm)'
time_s = 'Time (s)'
angle_deg = 'Angle (Degree)'
sample = 'Sample Number'
print_message_after_each_sample_loaded = False
sample_order = '날짜'
# 2 truncate & pad
chunk_size = 5000
chunk_size_mb = 50
sampling_rate_hz = 200
target_sample_length = 2048
peak_position = 1536
pad_value = 0.0
# 3 remove bad samples automatically
threshold_final_torque_nm_min = 5.0
threshold_final_torque_nm_max = 30.0
threshold_seat_torque_nm_min = 1.0
threshold_seat_torque_nm_max = 10.0
threshold_total_time_s_min = 1.0
threshold_total_time_s_max = 10.0
bad_sample_folder_name = 'Bad Samples Here'
# 4 remove bad samples manually
# 5 generate * labels from part spec
label_list_txt = 'Label List.txt'
data_label_info_txt = 'Data Label Info.txt'
# 6 prepare training
test_ratio = 0.1
valid_ratio = 0.1
train = 'Training'
valid = 'Validation'
test = 'Test'
data_type = 'Data Type'
data_split_info_txt = 'Data Split Info.txt'
mlp = 'MLP'
anns = [mlp]
seeds = range(1, 2)
mlp_act_f_s = ['relu']
mlp_k_init_s = ['GlorotNormal']
mlp_opt_s = ['adam']
#a, r, n = 0.001, 1/2, 2
#learning_rates = [a * (r ** i) for i in reversed(range(n))]
lr_s = [0.001]
#a, d, n = 0.0, 0.1, 10
#dropout_rates = [a + d * i for i in range(n)]
dr_s = [0.0]
#mlp_units = [[2 ** x] for x in range(12, 2, -1)]
mlp_units = [[1], [2], [1,2], [1,2,3], [1,2,3,4]]
ann = 'ANN'
h_layer = 'Hidden Layer'
h_l_f_u = 'Hidden Layer First Unit'
h_l_l_u = 'Hidden Layer Last Unit'
unit = 'Unit'
act_f = 'Activation Function'
k_init = 'Kernel Initializer'
opt = 'Optimizer'
lr = 'Learning Rate'
dr = 'Dropout Rate'
seed = 'Seed'
model_list_csv = 'Model List.csv'
# 7 train & evaluate model
n_processor = 2
lr_digit = 2
dr_digit = 1
seed_digit = 1
max_epoch = 20
epoch_digit = 3
init = 'Init'
best = 'Best'
final = 'Final'
model_save_type = 'keras'
model_summary_txt = 'model summary.txt'
n_batch = 24
threshold='Threshold'
loss ='Loss'
accuracy = 'Accuracy'
hamming_accuracy = 'Hamming Accuracy'
subset_accuracy = 'Subset Accuracy'
micro_precision = 'Micro Precision'
weighted_precision = 'Weighted Precision'
macro_precision = 'Macro Precision'
sample_precision = 'Sample Precision'
micro_recall = 'Micro Recall'
weighted_recall = 'Weighted Recall'
macro_recall = 'Macro Recall'
sample_recall = 'Sample Recall'
micro_f1 = 'Micro F1'
weighted_f1 = 'Weighted F1'
macro_f1 = 'Macro F1'
sample_f1 = 'Sample F1'
model_metrics = ['binary_accuracy']
save_training_dynamics = True
epoch = 'Epoch'
history = 'History'
acu_los = 'Accuracy / Loss'
prediction = 'Prediction'
score = 'Score'
early_stopping = True
es_patience = 10
reduce_lr = False
rlr_patience = 5
rlr_ratio = 0.2
rlr_min_lr = 0.00001
confusion_matrix_txt = 'Confusion Matrix.txt'
confusion_matrix_csv = 'Confusion Matrix.csv'
add_candidate_to_model_list = True
max_adding_model = 1
# 8 collect test score
collect_metric = [loss, hamming_accuracy, subset_accuracy, micro_f1, macro_f1]
hyperparameters = [ann, h_layer, unit, h_l_f_u, h_l_l_u, act_f, k_init, opt, lr, dr, seed]
score_csv = 'Score.csv'
# 9 plot hyperparameters 0612
common_plot_margin = (0.1, 0.97, 0.98, 0.1) # left, right, top, bottom
plot_score_margin_dict = {
    ann: [anns, 0.15, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    h_layer: [None, 0.1, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    h_l_f_u: [None, 0.15, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    h_l_l_u: [None, 0.15, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    unit: [mlp_units, 0.35, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    act_f: [mlp_act_f_s, 0.225, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    k_init: [mlp_k_init_s, 0.25, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    opt: [mlp_opt_s, 0.2, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    lr: [lr_s, 0.2, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    dr: [dr_s, 0.1, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
    seed: [seeds, 0.1, common_plot_margin[1], common_plot_margin[2], common_plot_margin[3]],
} # left, right, top, bottom
score_left_lims = [0.0, 0.4, 0.6, 0.8, 0.9]
integer_type_hyperparameters = [h_layer, h_l_f_u, h_l_l_u, seed]
float_type_hyperparameters = [lr, dr]
