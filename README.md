# Master's thesis data project

## Project overview
My master's thesis is focusing on the gamma imbalance of the option market makers' and its effect on the underlying market volatility in the option markets of Europe. 

## Roadmap
1. decompress.py - The original extension of the dataset is .lzip which resulting after decompress a very large sas7bdat file. This script provides a quick lzip decompress command for several lzip files in a directory.

2. sasdata_read_in.py - The decompressed file is sas7bdat. This extension is not supported by the most common parallel computing packages, therefore alternative data processing was necessary. This script is running through the data in chunks, selecting the relevant tickers via the key 'SecurityID' and saving the chunks in .parquet, which is a much more managable format. The provided data sample (data_sample.parquet) is one of the hundreds of the output files of the script after running.

3. raw_to_gex.py - Data processing. Combines the small pq. files into one dataset. Transforming the SAS date format into readable, assigning ticker names to security ID, aggregating gamma exposure (based on SqueezeMetrics and Sergei Perfiliev) across the option chain and grouping by date and ticker. As result we get the aggregation of the entire option chain: on one trading day, for one underlying equity index the aggregated gamma exposure. 

4. regression.py - Data preparation, exploration + Time series regression analysis

   demonstration.ipynb - Jupyter Notebook for demonstration purposes. More readable than the .py regression analysis. Containing only DAX, the other indices are in the .py file. 

5. Horvath_MThesis.pdf - The draft version of the thesis in Hungarian.

## Acknowledgements
Special thanks to Péter Gönczi and Áron Polgár who helped me with the technical implementation. 
