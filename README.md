# Master's thesis data project

## Project overview
My master's thesis is focusing on the gamma imbalance of the option market makers' and its effect on the underlying market volatility in the option markets of Europe. 

***Key findings:***
>a significant and robust negative relationship between the net gamma exposure of option market makers and the volatility of the underlying product for Europen equity indices with higher option volumes (DAX, FTSE-100, SMI)

>for indices with smaller option volumes, no relationship can be detected (HEX & BEL20)

>according to the regression estimates: one standard deviation increase in the net gamma exposure results in average 8.94-12.22 bps. decrease in absolute log returns in the underlying market 

>the *gamma effect* is weaker than observed in the U.S. markets, which can be explained with differences in market structure and liquidity, and in the overall importance of options trading within the two regions derivatives market 

<p align="center">
  <img src="https://github.com/balint41/Thesis/blob/main/heatm.png" alt="the evolution of realised volatility as a function of net gamma exposure (DAX)"/>
</p>
   
## Roadmap
1. decompress.py - The original extension of the dataset is .lzip which resulting after decompress a very large sas7bdat file. This script provides a quick lzip decompress command for several lzip files in a directory.

2. sasdata_read_in.py - The decompressed file is sas7bdat. This extension is not supported by the most common parallel computing packages, therefore alternative data processing was necessary. This script is running through the data in chunks, selecting the relevant tickers via the key 'SecurityID' and saving the chunks in .parquet, which is a much more managable format. The provided data sample (data_sample.parquet) is one of the hundreds of the script's output files after running.

3. raw_to_gex.py - Data processing. Combines the small pq. files into one dataset. Transforming the SAS date format into readable, assigning ticker names to security ID, aggregating gamma exposure across the option chain and grouping by date and ticker. As result we get the aggregation of the entire option chain: on one trading day, for one underlying equity index the aggregated gamma exposure. 

4. regression.py - Data preparation, exploration + Time series regression analysis

   demonstration.ipynb - Jupyter Notebook for demonstration purposes. More readable than the .py regression analysis. Containing only DAX, the other indices, and the full analysis is in the .py file. 

5. Thesis.pdf - The final version of the thesis in Hungarian.
 
