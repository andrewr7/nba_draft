import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

import cairosvg
from io import BytesIO

BIN_SIZE = 10
plt.rcParams['font.size'] = 14

def get_team_logo(team, scale=0.8):
    with open(f'logos/{team}.svg', 'rb') as f:
        svg_data = f.read()
    png_data = cairosvg.svg2png(bytestring=svg_data,scale=scale)
    image_stream = BytesIO(png_data)
    img = mpimg.imread(image_stream, format='png')
    return img

alleged_odds_2019_present = np.array([14.0, 14.0, 14.0, 12.5, 10.5, 9.0, 7.5, 6.0, 4.5, 3.0, 2.0, 1.5, 1.0, 0.5]) / 100
#Source for this years odds: https://sports.yahoo.com/nba/article/nba-draft-lottery-2025-date-time-no-1-odds-and-what-to-know-for-the-cooper-flagg-sweepstakes-171018514.html
alleged_odds_2025_modified = np.array([14.0, 14.0, 14.0, 12.5, 10.5, 9.0, 7.5, 6.0, 3.8, 3.7, 1.8, 1.7, 0.8, 0.7]) / 100

years_dict = {
    2025: {
        'alleged_odds': alleged_odds_2025_modified,
        'team_odds_order': ['Jazz', 'Wizards', 'Hornets', 'Pelicans', 'Sixers', 'Nets', 'Raptors', 'Spurs', 'Suns', 'Blazers', 'Mavericks', 'Bulls', 'Kings', 'Hawks'],
        'actual_draft_lottery_outcome_order': np.array([11,8,5,3]) - 1, #subtract 1 to zero-index
    },
}

for year_of_interest in years_dict:
    assert np.isclose(sum(years_dict[year_of_interest]['alleged_odds']),1.0)
    alleged_odds_indices = np.arange(len(years_dict[year_of_interest]['alleged_odds']))
    for num_picks_included in [1,2,3,4]:
    # for num_picks_included in [3]:
        perms = list(permutations(alleged_odds_indices, num_picks_included))
        num_outcomes = len(perms)
        perm_dict_list = [{'indices':p, 'odds_list':[years_dict[year_of_interest]['alleged_odds'][idx] for idx in p]} for p in permutations(alleged_odds_indices, num_picks_included)]
        cumulative_likelihood_percent = 0.0
        for perm_dict in perm_dict_list:        
            likelihood = 1.0
            cumulative_denominator = 1.0
            for pick_odds in perm_dict['odds_list']:
                likelihood *= pick_odds/cumulative_denominator
                cumulative_denominator -= pick_odds
            perm_dict['likelihood'] = likelihood
            perm_dict['likelihood_percent'] = likelihood*100
            cumulative_likelihood_percent += perm_dict['likelihood_percent']

        assert np.isclose(cumulative_likelihood_percent, 100.0), cumulative_likelihood_percent
        permutations_df = pd.DataFrame(perm_dict_list)
        permutations_df = permutations_df.sort_values(by='likelihood').reset_index(drop=True).reset_index() #add an additional col called index for convenience
        permutations_df['cdf'] = permutations_df['likelihood_percent'].cumsum()
        matches = permutations_df[permutations_df['indices'] == tuple(years_dict[year_of_interest]['actual_draft_lottery_outcome_order'][:num_picks_included])]
        assert not matches.empty
        actual_order_index = matches.index[0]
        

        # Define bins — here deciles
        thresholds = np.linspace(BIN_SIZE, 100, BIN_SIZE)
        permutations_df['x_label'] = None
        permutations_df['y_label'] = None
        for t in thresholds:
            try:
                idx = permutations_df[permutations_df['cdf'] >= t].index[0]
                permutations_df.at[idx, 'x_label'] = f'{t:.0f}%'
                permutations_df.at[idx, 'y_label'] = f"{permutations_df.at[idx, 'likelihood_percent']:.2g}%"
            except:
                if t == thresholds[-1]:
                    permutations_df.at[num_outcomes-1, 'x_label'] = f'{t:.0f}%'
                    permutations_df.at[num_outcomes-1, 'y_label'] = f"{permutations_df.at[num_outcomes-1, 'likelihood_percent']:.2g}%"

        filtered_df = permutations_df[permutations_df['x_label'].notna()]

        # print_list = '['
        for i, row in filtered_df.iterrows():
            teams = []
            for team_index in row['indices']:
                teams.append(years_dict[year_of_interest]['team_odds_order'][team_index])
            print(f"Cumulative {row['x_label']} probability permutation: {teams}")
            # print_list = print_list + f"{row['likelihood']}, "
        # print_list = print_list + ']'
        # print(print_list)
        # continue
        

        highlight_indices = {actual_order_index:'red'}
        colors = [highlight_indices[i] if i in highlight_indices else 'skyblue' for i in permutations_df.index]
        valid = permutations_df[permutations_df['x_label'].notnull()]

        fig, ax = plt.subplots()
        ax.margins(x=0.01, y=0.01)
        bars = ax.bar(permutations_df['index'], permutations_df['likelihood_percent'], color=colors)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xticks(valid['index'])
        ax.set_xticklabels(valid['x_label'], rotation=315, fontsize=10)
        ax.set_yticks(valid['likelihood_percent'])
        ax.set_yticklabels(valid['y_label'])
        ax.set_xlabel('Cumulative Probability Deciles')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Outcome Probability')
        if num_picks_included == 4:
            ax.set_title(f'{year_of_interest} NBA Draft Lottery: All {num_outcomes} Possible Outcomes\n', fontsize=20)
        else:
            ax.set_title(f'{year_of_interest} NBA Draft Lottery: All {num_outcomes} Possible Outcomes for the First {num_picks_included} Picks\n', fontsize=16)
        ax.grid()


        # Add a second x-axis on bottom
        secax = ax.secondary_xaxis('bottom')

        permutations_df['x_label2'] = None
        permutations_df.at[actual_order_index, 'x_label2'] = f"\n\nActual Outcome: {permutations_df.at[actual_order_index, 'likelihood_percent']:.2g}%\n{permutations_df.at[actual_order_index, 'cdf']:.2g}% Cumulative Probability"
        for i, team_index in enumerate(permutations_df.at[actual_order_index, 'indices']):
            imagebox = OffsetImage(get_team_logo(team=years_dict[year_of_interest]['team_odds_order'][team_index]), zoom=0.1)
            x_coord = (actual_order_index + 0.5)/num_outcomes + 0.035*(i + 0.5 - num_picks_included/2)
            ab = AnnotationBbox(imagebox, (x_coord, -0.05), xycoords='axes fraction', frameon=False)  # Place at data coords
            ax.add_artist(ab)
        valid2 = permutations_df[permutations_df['x_label2'].notnull()]
        secax.tick_params(axis='x', color=highlight_indices[actual_order_index], labelcolor=highlight_indices[actual_order_index], labelsize=12, length=8,width=2,direction='out')
        secax.set_xticks(valid2['index'])              # Custom tick positions
        secax.set_xticklabels(valid2['x_label2'])  # Custom labels



        # Add a third x-axis on bottom
        thirdax = ax.secondary_xaxis('bottom')

        permutations_df['x_label3'] = None
        permutations_df.at[0, 'x_label3'] = f"\n\nLeast Probable: {permutations_df.at[0, 'likelihood_percent']:.2g}%"
        for i, team_index in enumerate(permutations_df.at[0, 'indices']):
            imagebox = OffsetImage(get_team_logo(team=years_dict[year_of_interest]['team_odds_order'][team_index]), zoom=0.1)
            x_coord = 0.5/num_outcomes + 0.035*(i + 0.5 - num_picks_included/2)
            ab = AnnotationBbox(imagebox, (x_coord, -0.05), xycoords='axes fraction', frameon=False)  # Place at data coords
            ax.add_artist(ab)

        permutations_df.at[num_outcomes-1, 'x_label3'] = f"\n\nMost Probable: {permutations_df.at[num_outcomes-1, 'likelihood_percent']:.2g}%"
        for i, team_index in enumerate(permutations_df.at[num_outcomes-1, 'indices']):
            imagebox = OffsetImage(get_team_logo(team=years_dict[year_of_interest]['team_odds_order'][team_index]), zoom=0.1)
            # x_coord = 1 - 1/num_outcomes/2
            x_coord = 1 - 0.5/num_outcomes + 0.035*(i + 0.5 - num_picks_included/2)
            ab = AnnotationBbox(imagebox, (x_coord, -0.05), xycoords='axes fraction', frameon=False)  # Place at data coords
            ax.add_artist(ab)
        
        valid3 = permutations_df[permutations_df['x_label3'].notnull()]

        thirdax.tick_params(axis='x', color='blue', labelsize=12, labelcolor='blue',length=8,width=2,direction='out')
        thirdax.set_xticks(valid3['index'])      
        thirdax.set_xticklabels(valid3['x_label3'])

        width_px, height_px = 1920, 1080
        dpi = 150
        fig.set_size_inches(width_px / dpi, height_px / dpi)
        fig.savefig(f'Lottery_{year_of_interest}_odds_first{num_picks_included}.png', dpi=dpi, bbox_inches='tight')
        
