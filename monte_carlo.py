import numpy as np
import pandas as pd

###### set run parameters
# MAX_PICKS_TO_ANALYZE = 7 #how many picks per draft to analyze, e.g. first pick only, first 2, first 3, etc
MAX_PICKS_TO_ANALYZE = 2 #how many picks per draft to analyze, e.g. first pick only, first 2, first 3, etc
# YEARS_TO_ANALYZE = range(1990,2026) #default to every year since weighted lottery began
YEARS_TO_ANALYZE = range(2003,2026)
NUM_ITERS_TO_SIMULATE = 500
######
throwout_years = [1995, 1996, 1997, 1998, 2004]

lottery_picks_per_year = {}
for year in range(2019, 2026):
    lottery_picks_per_year[year] = 4

for year in range(1987, 2019):
    lottery_picks_per_year[year] = 3

for year in range(1985, 1987):
    lottery_picks_per_year[year] = 7

def modify_odds(selected_index, input_odds):
    assert np.isclose(np.sum(input_odds),1.0)
    selected_probability = input_odds[selected_index]
    new_odds = input_odds.copy()
    new_odds[selected_index] = 0.0
    new_odds = new_odds/(1.0 - selected_probability)
    assert np.isclose(np.sum(new_odds),1.0)
    return new_odds

def calculate_actual_outcome_odds(years_dict, num_picks_to_analyze):
    actual_outcome_odds_list = []
    for year in years_dict:
        current_odds = years_dict[year]['alleged_odds']
        num_picks_to_loop = min(num_picks_to_analyze, len(years_dict[year]['actual_draft_order']))
        for pick_idx in range(num_picks_to_loop):
            selected_index = years_dict[year]['actual_draft_order'][pick_idx]
            actual_outcome_odds_list.append(current_odds[selected_index])
            current_odds = modify_odds(selected_index, current_odds)
    return np.array(actual_outcome_odds_list)

draft_lottery_history_df = pd.read_csv('nba_draft_lottery_results_1985_2025.csv')
years_dict = {}
for year in YEARS_TO_ANALYZE:
    if MAX_PICKS_TO_ANALYZE > 1:
        if year in throwout_years: #throw out years with expansion weirdness, unless only analyzing first pick
            continue
    this_df = draft_lottery_history_df[draft_lottery_history_df['Year'] == year].copy()
    if MAX_PICKS_TO_ANALYZE == 1:
        #with expansion weirdness, have to calculate odds for first pick this way
        this_df['Odds_float'] = this_df['Odds'].str.replace('%', '').astype(float)
        this_df['Odds_decimal'] = this_df['Odds_float'] / 100
    else:
        #this is a better way to calculate odds if you can
        this_df['Odds_decimal'] = this_df['Chances'] / this_df['Chances'].sum()
    this_df['Position_num'] = this_df['Pre-Lottery Position'].str.extract('(\d+)').astype(int)

    #sort by odds first
    this_df = this_df.sort_values(by='Position_num').reset_index(drop=True)
    alleged_odds = this_df['Odds_decimal'].to_numpy()
    if MAX_PICKS_TO_ANALYZE > 1:
        assert np.all(alleged_odds[:-1] >= alleged_odds[1:]), alleged_odds #make sure its sorted descending or else something went wrong
        assert np.isclose(np.sum(alleged_odds), 1.0), np.sum(alleged_odds)
    else:
        #some numerical error with this method that may need to be corrected
        if not np.isclose(np.sum(alleged_odds), 1.0):
            assert np.abs(np.sum(alleged_odds) - 1.0) < 0.01
            alleged_odds = alleged_odds / np.sum(alleged_odds)
    team_odds_order = list(this_df['Team'])

    # then sort by actual pick
    this_df = this_df.sort_values(by='Pick')
    actual_draft_order = this_df['Position_num'].to_numpy() - 1 #zero-index
    actual_draft_order = actual_draft_order[:lottery_picks_per_year[year]] #only take as many picks are actually from the lottery

    years_dict[year] = {
        'alleged_odds': alleged_odds,
        'team_odds_order':team_odds_order,
        'actual_draft_order':actual_draft_order,
    }


simulated_draft_dict_list = []
simulated_odds_only_list = []
actual_outcome_odds = calculate_actual_outcome_odds(years_dict, MAX_PICKS_TO_ANALYZE)
total_picks_per_iter = actual_outcome_odds.shape[0]
for i in range(NUM_ITERS_TO_SIMULATE):
    if i%1000 == 0:
        print(f'-- Running simulated draft iteration #{i} --')
    total_odds_list_this_iter = []
    for year in years_dict:
        simulated_draft_dict = {'iter':i, 'year':year, 'teams':[], 'indices':[], 'probabilities':[], 'cumulative_probabilities':[]}
        current_odds = years_dict[year]['alleged_odds']
        num_picks_to_loop = min(MAX_PICKS_TO_ANALYZE, len(years_dict[year]['actual_draft_order']))
        for pick_idx in range(num_picks_to_loop):
            selected_index = np.random.choice(np.arange(current_odds.shape[0]), p=current_odds)
            simulated_draft_dict['teams'].append(years_dict[year]['team_odds_order'][selected_index])
            simulated_draft_dict['indices'].append(selected_index)
            simulated_draft_dict['probabilities'].append(current_odds[selected_index])
            if len(simulated_draft_dict['cumulative_probabilities']):
                simulated_draft_dict['cumulative_probabilities'].append(simulated_draft_dict['cumulative_probabilities'][-1]*current_odds[selected_index])
            else:
                simulated_draft_dict['cumulative_probabilities'].append(current_odds[selected_index])
            total_odds_list_this_iter.append(current_odds[selected_index])
            current_odds = modify_odds(selected_index, current_odds)
        simulated_draft_dict_list.append(simulated_draft_dict)
    assert total_picks_per_iter == len(total_odds_list_this_iter)
    simulated_odds_only_list.append(np.array(total_odds_list_this_iter))

simulated_drafts_df = pd.DataFrame(simulated_draft_dict_list)
simulated_odds_np = np.array(simulated_odds_only_list)


log_likelihoods = np.sum(np.log(simulated_odds_np), axis=1)
log_likelihood_obs = np.sum(np.log(actual_outcome_odds))
p_value = np.sum(log_likelihoods <= log_likelihood_obs)/log_likelihoods.shape[0]
print(f"Results for years {[year for year in YEARS_TO_ANALYZE if not year in throwout_years]}")
print(f"All lottery picks up to pick {MAX_PICKS_TO_ANALYZE} in each draft")
if MAX_PICKS_TO_ANALYZE > 1:
    throwout_years_filtered = [year for year in throwout_years if year in YEARS_TO_ANALYZE]
    if throwout_years_filtered:
        print(f"Note that years {throwout_years_filtered} were thrown out due to expansion team weirdness")
print('Total number of historical picks analyzed:', total_picks_per_iter)
print('Number of simulation iterations run:', NUM_ITERS_TO_SIMULATE)
print('p_value:', p_value)
print(f"Assuming a fair draft lottery, there's a {p_value*100:.4g}% chance we would see a sequence of outcomes this unlikely (or even more unlikely) for these {total_picks_per_iter} draft picks.")

# #only valid for 2025
# pick_decile_bins = {
#     1:[0.038, 0.075, 0.09, 0.105, 0.125, 0.14, 0.14, 0.14, ],
#     2:[0.002767441860465117, 0.0053790238836967826, 0.007978723404255319, 0.010384615384615383, 0.012209302325581397, 0.014651162790697675, 0.017093023255813956, 0.02, 0.02279069767441861, 0.02279069767441861, ],
#     3:[0.00022857999816168473, 0.0004675996784226773, 0.0007684591257170311, 0.001034013605442177, 0.0013997926233150645, 0.0018075400241005337, 0.002247752247752248, 0.002727272727272727, 0.003323643410852714, 0.004431524547803619, ],
#     4:[2.2775078931663338e-05, 4.90424178834697e-05, 8.16326530612245e-05, 0.00012120553979859787, 0.00017233560090702948, 0.00023689475790316132, 0.0003171247357293869, 0.00041218885758181754, 0.0005919661733615223, 0.0009550699456473318, ],
# }
# for num_picks in pick_decile_bins:
#     data = np.array(simulated_drafts_df['cumulative_probabilities'].apply(lambda x: x[num_picks-1]))
#     bin_edges = pick_decile_bins[num_picks]

#     # Create histogram
#     plt.hist(data, bins=bin_edges, edgecolor='black')

#     # Add labels and title
#     plt.xlabel('Bin deciles')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram for {num_picks} picks')

#     # Show plot
#     plt.show()