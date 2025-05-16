import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

###### run parameters ######

# sim parameters
SIM_FOLDER = 'sims_1985to2025_1M_iters' # use an existing folder name to read in data, new name to create new data
ALL_AVAILABLE_YEARS = list(range(1985,2026)) # Every year the lottery has existed
NUM_ITERS_TO_SIMULATE = 1000000 #one million is a lot. consider reducing

# analysis parameters
ALLEGED_RIGGED_YEARS = [2003,2008,2011,2012,2014,2019,2025] #years that people claim are rigged for one reason or another
ALLEGED_NONRIGGED_YEARS = [year for year in ALL_AVAILABLE_YEARS if year not in ALLEGED_RIGGED_YEARS]
YEAR_SUBSETS = [ALL_AVAILABLE_YEARS, ALLEGED_RIGGED_YEARS, ALLEGED_NONRIGGED_YEARS]
PICK_INDICES_SUBSETS = [[0], [1], [2], [1,2], [1,2,3]] # which pick subsets to analyze. be careful, these are zero-indexed

######

def get_draft_history_dict():
    """
        read in dataframe with draft lottery data. odds, outcomes, etc. and organize the data into a form we can use

    """
    #not every year had the same number of lottery picks.
    lottery_picks_per_year = {}
    for year in range(2019, 2026):
        lottery_picks_per_year[year] = 4

    for year in range(1987, 2019):
        lottery_picks_per_year[year] = 3

    for year in range(1985, 1987):
        lottery_picks_per_year[year] = 7
        
    draft_lottery_history_df = pd.read_csv('nba_draft_lottery_results_1985_2025.csv')
    draft_lottery_history_dict = {}
    for year in ALL_AVAILABLE_YEARS:
        this_df = draft_lottery_history_df[draft_lottery_history_df['Year'] == year].copy()
        this_df['Odds_decimal'] = this_df['Chances'] / this_df['Chances'].sum()
        this_df['Position_num'] = this_df['Pre-Lottery Position'].str.extract('(\d+)').astype(int)

        #sort by odds first
        this_df = this_df.sort_values(by=['Odds_decimal', 'Position_num'], ascending=[False,True]).reset_index(drop=True)
        alleged_odds = this_df['Odds_decimal'].to_numpy()
        assert np.all(alleged_odds[:-1] >= alleged_odds[1:]), alleged_odds #make sure its sorted descending or else something went wrong
        assert np.isclose(np.sum(alleged_odds), 1.0), np.sum(alleged_odds)
        team_odds_order = list(this_df['Team'])

        # then sort by actual pick
        this_df = this_df.sort_values(by='Pick')
        actual_draft_lottery_outcome_order = this_df['Position_num'].to_numpy() - 1 #zero-index
        actual_draft_lottery_outcome_order = actual_draft_lottery_outcome_order[:lottery_picks_per_year[year]] #only take as many picks are actually from the lottery

        draft_lottery_history_dict[year] = {
            'alleged_odds': alleged_odds,
            'team_odds_order':team_odds_order,
            'actual_draft_lottery_outcome_order':actual_draft_lottery_outcome_order,
        }

    # Due to a very bizarre agreement related to expansion teams, the 1st and 2nd picks in 1996 and 1998 were flipped from the actual lottery outcome. flipping them back to analyze the actual lottery outcome.
    for weird_year in [1996, 1998]:
        if weird_year in draft_lottery_history_dict:
            draft_lottery_history_dict[weird_year]['actual_draft_lottery_outcome_order'][0], draft_lottery_history_dict[weird_year]['actual_draft_lottery_outcome_order'][1] \
            = draft_lottery_history_dict[weird_year]['actual_draft_lottery_outcome_order'][1], draft_lottery_history_dict[weird_year]['actual_draft_lottery_outcome_order'][0]

    return draft_lottery_history_dict


def modify_odds(selected_index, input_odds):
    """
        modify the sampling distribution for pick n+1 given the result of pick n
    """
    assert np.isclose(np.sum(input_odds),1.0)
    selected_probability = input_odds[selected_index]
    new_odds = input_odds.copy()
    new_odds[selected_index] = 0.0
    new_odds = new_odds/(1.0 - selected_probability)
    assert np.isclose(np.sum(new_odds),1.0) or np.isclose(np.sum(new_odds), 0.0)
    return new_odds


def calculate_observed_outcome_probs(draft_lottery_history_dict):
    """
        calculate the probabilities of the picks that actually happened in reality
    """
    draft_dict_list = []
    for year in draft_lottery_history_dict:
        current_odds = draft_lottery_history_dict[year]['alleged_odds']
        cum_prob = 1.0
        for pick_idx in range(len(draft_lottery_history_dict[year]['actual_draft_lottery_outcome_order'])):
            assert np.isclose(np.sum(current_odds), 1.0)
            selected_index = draft_lottery_history_dict[year]['actual_draft_lottery_outcome_order'][pick_idx]
            this_prob = current_odds[selected_index]
            cum_prob *= this_prob
            team_name = draft_lottery_history_dict[year]['team_odds_order'][selected_index]
            draft_dict_list.append({'iter':-1, 'year':year, 'pick_idx':pick_idx, 'probability': this_prob, 'cum_prob': cum_prob, 'team':team_name})
            current_odds = modify_odds(selected_index, current_odds)

    return pd.DataFrame(draft_dict_list)


def generate_simulated_draft_data(draft_lottery_history_dict, num_iters):
    sim_probs_dict_list = []
    for i in range(num_iters):
        if i%1000 == 0:
            print(f'Running simulated draft iteration #{i}')

        for year in draft_lottery_history_dict:
            current_odds = draft_lottery_history_dict[year]['alleged_odds']
            index_array_to_sample = np.arange(current_odds.shape[0])
            cum_prob = 1.0

            # simulate the same number of lottery picks as there actually were in any given year
            for pick_idx in range(len(draft_lottery_history_dict[year]['actual_draft_lottery_outcome_order'])):
                assert np.isclose(np.sum(current_odds), 1.0)

                # simulate a lottery pick here
                selected_index = np.random.choice(index_array_to_sample, p=current_odds)
                this_prob = current_odds[selected_index]
                cum_prob *= this_prob
                team_name = draft_lottery_history_dict[year]['team_odds_order'][selected_index]
                sim_probs_dict_list.append({'iter':i, 'year':year, 'pick_idx':pick_idx, 'probability': this_prob, 'cum_prob': cum_prob, 'team':team_name})
                # update odds for next pick
                current_odds = modify_odds(selected_index, current_odds)
    
    return pd.DataFrame(sim_probs_dict_list)


def plot_histograms_2025(sim_probs_df):
    """
        This function isnt really relevant to the rest of the script its just a sanity check for the monte carlo simulations to make sure the output distributions make sense.
        Each decile bin should be filled roughly equally, mostly for the num_picks = 3 and 4 plots. 1 and 2 the discreteness of the data makes it whacky
        It is ony valid for 2025, becuase thats the only year I have calculated decile bins for.
    """
    pick_decile_bins = {
        # 1:[0.038, 0.075, 0.09, 0.105, 0.125, 0.14, 0.14, 0.14, ],
        # 2:[0.002767441860465117, 0.0053790238836967826, 0.007978723404255319, 0.010384615384615383, 0.012209302325581397, 0.014651162790697675, 0.017093023255813956, 0.02, 0.02279069767441861, 0.02279069767441861, ],
        3:[0.00022857999816168473, 0.0004675996784226773, 0.0007684591257170311, 0.001034013605442177, 0.0013997926233150645, 0.0018075400241005337, 0.002247752247752248, 0.002727272727272727, 0.003323643410852714, 0.004431524547803619, ],
        4:[2.2775078931663338e-05, 4.90424178834697e-05, 8.16326530612245e-05, 0.00012120553979859787, 0.00017233560090702948, 0.00023689475790316132, 0.0003171247357293869, 0.00041218885758181754, 0.0005919661733615223, 0.0009550699456473318, ],
    }
    filtered_df = sim_probs_df[sim_probs_df['year'] == 2025].copy()
    for num_picks in pick_decile_bins:
        data = np.array(filtered_df[filtered_df['pick_idx'] == num_picks-1]['cum_prob'])
        bin_edges = pick_decile_bins[num_picks]

        # Create histogram
        plt.hist(data, bins=bin_edges, edgecolor='black')

        # Add labels and title
        plt.xlabel('Bin deciles')
        plt.ylabel('Frequency')
        plt.title(f'Histogram for {num_picks} picks')

        # Show plot
        plt.show()


def main():
    # read in data from file if it already exists, else run the simulation
    sim_data_path = os.path.join(SIM_FOLDER, 'sim_data.pickle')
    if os.path.isfile(sim_data_path):
        print(f"Loading simulated data from {sim_data_path}")
        with open(sim_data_path, 'rb') as f:
            draft_lottery_history_dict, obs_probs_df, sim_probs_df = pickle.load(f)
    else:
        print(f"Generating simulated data")
        draft_lottery_history_dict = get_draft_history_dict()
        obs_probs_df = calculate_observed_outcome_probs(draft_lottery_history_dict)
        sim_probs_df = generate_simulated_draft_data(draft_lottery_history_dict, NUM_ITERS_TO_SIMULATE)

        plot_histograms_2025(sim_probs_df) # sanity check for the generated data
        print(f"Writing simulated data to {sim_data_path}")
        if not os.path.isdir(SIM_FOLDER):
            os.mkdir(SIM_FOLDER)
        with open(sim_data_path, 'wb') as f:
            pickle.dump((draft_lottery_history_dict, obs_probs_df, sim_probs_df), f)
        #write to csv for manual viewing later if so desired
        obs_probs_df.to_csv(os.path.join(SIM_FOLDER, 'obs_data.csv'), index=False)
        sim_probs_df.to_csv(os.path.join(SIM_FOLDER, 'sim_data.csv'), index=False)


    #make sure sim_probs_df meets certain necessary criteria
    assert sim_probs_df['iter'].is_monotonic_increasing, "'iter' column is not sorted in increasing order"
    group_sizes = sim_probs_df.groupby('iter').size()
    assert group_sizes.nunique() == 1, f"'iter' groups are not evenly sized: found sizes {group_sizes.unique()}"

    sim_iter_list = sim_probs_df['iter'].unique().tolist()
    num_sim_iters = len(sim_iter_list)
    assert num_sim_iters - 1 == max(sim_iter_list)

    years_simulated_list = sim_probs_df['year'].unique().tolist()

    print("-------------------------------------------------------------------------------\n")
    print(f'Draft Lottery Monte Carlo Analysis: actual outcome compared with {num_sim_iters} simulations\n')

    #segregate into subsets of years and picks to analyze
    for year_subset in YEAR_SUBSETS:
        years_to_analyze = [year for year in year_subset if year in years_simulated_list]
        print(f"\n-- RESULTS FOR DRAFT YEARS {years_to_analyze} --\n")
        obs_probs_filtered_year_df = obs_probs_df[obs_probs_df['year'].isin(years_to_analyze)].copy()
        sim_probs_filtered_year_df = sim_probs_df[sim_probs_df['year'].isin(years_to_analyze)].copy()
        for pick_indices_subset in PICK_INDICES_SUBSETS:
            obs_probs_filtered_year_picks_df = obs_probs_filtered_year_df[obs_probs_filtered_year_df['pick_idx'].isin(pick_indices_subset)].copy()
            sim_probs_filtered_year_picks_df = sim_probs_filtered_year_df[sim_probs_filtered_year_df['pick_idx'].isin(pick_indices_subset)].copy()
            # filter real observations for the year and pick indices of interest
            filtered_obs_probs_np = np.array(obs_probs_filtered_year_picks_df['probability'])
            tot_filtered_picks = filtered_obs_probs_np.shape[0]
            log_likelihood_obs_np = np.sum(np.log(filtered_obs_probs_np))

            #filter each simulation for the year and pick indices of interest
            #assumes that iters are sorted and evenly sized, which should be the case
            filtered_sim_probs_np = sim_probs_filtered_year_picks_df['probability'].values.reshape(num_sim_iters, -1)
            log_likelihoods_sim_np = np.sum(np.log(filtered_sim_probs_np), axis=1)

            p_value = np.sum(log_likelihoods_sim_np <= log_likelihood_obs_np)/log_likelihoods_sim_np.shape[0]
            print(f"Pick {' and '.join([str(pick_idx + 1) for pick_idx in pick_indices_subset])} only in each draft ({tot_filtered_picks} picks across {len(years_to_analyze)} drafts):")
            print('\tp_value:', p_value)
            print(f"\tIf the lottery system is fair, there's a {p_value*100:.4g}% chance of seeing a result this unusual (or even more unusual) just by random chance.\n")


        # # Repeat process above but don't filter any picks
        filtered_obs_probs_np = np.array(obs_probs_filtered_year_df['probability'])
        tot_filtered_picks = filtered_obs_probs_np.shape[0]
        log_likelihood_obs_np = np.sum(np.log(filtered_obs_probs_np))
        
        #assumes that iters are sorted and evenly sized, which should be the case
        filtered_sim_probs_np = sim_probs_filtered_year_df['probability'].values.reshape(num_sim_iters, -1)
        log_likelihoods_sim_np = np.sum(np.log(filtered_sim_probs_np), axis=1)

        p_value = np.sum(log_likelihoods_sim_np <= log_likelihood_obs_np)/log_likelihoods_sim_np.shape[0]
        print(f"All lottery picks in each draft ({tot_filtered_picks} picks across {len(years_to_analyze)} drafts):")
        print('\tp_value:', p_value)
        print(f"\tIf the lottery system is fair, there's a {p_value*100:.4g}% chance of seeing a result this unusual (or even more unusual) just by random chance.\n")

if __name__ == "__main__":
    main()
