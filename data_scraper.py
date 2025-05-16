#i didnt write this one its all chatgpt. kinda crazy
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the base URL for the draft lottery results
base_url = "https://basketball.realgm.com/nba/draft/lottery_results/"

# Initialize an empty list to store the data
all_data = []

# Loop through each year from 1985 to 2025
for year in range(1985, 2026):
    url = f"{base_url}{year}"
    print(f"Scraping {url}")
    
    # Send a GET request to fetch the page content
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table containing the draft lottery results
        table = soup.find('table', {'class': 'table table-striped table-centered table-hover table-bordered table-compact table-nowrap'})
        
        # Check if the table exists
        if table:
            # Extract the header row
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            
            # Extract the data rows
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip the header row
                cells = tr.find_all('td')
                if len(cells) > 0:
                    row = [cell.get_text(strip=True) for cell in cells]
                    rows.append(row)
            
            # Create a DataFrame for the current year
            df = pd.DataFrame(rows, columns=headers)
            df['Year'] = year  # Add a column for the year
            
            # Append the DataFrame to the list
            all_data.append(df)
        else:
            print(f"No table found for {year}")
    else:
        print(f"Failed to retrieve data for {year}")

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# Display the first few rows of the final DataFrame
print(final_df.head())

final_df.to_csv('nba_draft_lottery_results_1985_2025.csv', index=False)