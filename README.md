# 1.Business Question

In the fast-paced and competitive realm of real estate, identifying and capitalizing on the best business opportunities is crucial. House Rocket, an (fictitious) innovative digital platform, is leading this challenge by harnessing advanced technology to transform the process of buying and selling properties.

The primary strategy is straightforward yet effective: the goal is to purchase **high-quality houses** located in **prime locations** at **low prices**, and subsequently resell them at **higher prices**. The larger the difference between the buying and selling price, the greater the profit margin, resulting in increased revenue for the company.

However, navigating the real estate market is a complex task. House prices are influenced by a multitude of factors. The location of a property can significantly impact its value - a house in a desirable neighborhood or one that's close to amenities like schools, parks, and shopping centers can often command a higher price. The time of year when the property is sold can also play a role, with certain seasons attracting more buyers.

House Rocket leaves nothing to chance. It utilizes technology to analyze these factors comprehensively. Machine learning algorithms are employed to predict house prices based on historical data and current market trends. The use of data visualization tools enables the identification of patterns and insights that might be overlooked in a traditional analysis.

This blend of real estate expertise and technological innovation sets House Rocket apart. It enables data-driven decisions, identification of lucrative business opportunities, and maintaining a competitive edge in the real estate market.

In order to optimize the process of buying and selling properties, the CEO requested the creation of a tool that he could access from anywhere in the world and could make his decisions based on the performance presented in this tool. So a number of analyses were drawn up and made available at [House Rocket App](https://house-rocket-company.streamlit.app/) so that he could follow the business metrics in real time. He would like to see the following metrics:

## On Overview:
  1. General behavioral metrics.
  2. Geolocation distribution.

## On EDA:
  1. Overall Metrics.
  2. Region Overview.
  3. Commercial Attributes
  4. Houses Attributes
  5. Price disribution
  6. Correlation between the variables
  7. Average price per selected feature

## On Hypotheses:
  1. Business team hypotheses.

## On Business questions:
  1. Solution of the business problem presented by the CEO.


# 2. Assumptions made for the analysis
  1. The analysis was carried out using data between 2014/05/02 and 2015/05/27.
  2. Properties with no bedrooms, no bathrooms and an unreasonable number of bathrooms (33 bathrooms) have been removed from the dataset.
  3. Some features not included in the original list of features were created to improve the analysis.

# 3. Solution Strategy
The app was developed using metrics that reflect the main visions of the company's business model.

# 4. Top 3 Data Insights
  1. The prices of properties with a water view are, on average, much higher than those without a water view.
  2. Winter is the best window for portfolio acquisitions, while early spring is the best period for sales. Otherwise, deals tend to be quick, requiring a lot of accuracy from the data analysis team to suggest the best properties for negotiations.
  3. It is possible to make a substantial profit by purchasing properties with the characteristics condition>=3 / grade>=5 in the low price season and selling them in the high price season. The result was a profit of $773,654,428.75 for House Rocket.

# 5. The final product of the project
Online App, hosted in a Cloud and available for access from any internet connected device.
To access the final result, please go to the [Streamlit App link](https://house-rocket-company.streamlit.app/).

# 6. Conclusion
The aim of this project was to create an App that presents insights in an effective and intuitive way for the CEO. Through the strategic use of technology and data analysis, it was possible to identify lucrative business opportunities in the King County real estate market. 

The app not only provides an overview of the market, but also offers insights into factors that influence real estate prices. This allows House Rocket to make informed decisions and maximize its profits.

In addition, the app demonstrates the power of data analysis and technology in transforming the real estate sector. It highlights how House Rocket is at the forefront of this transformation, using technology to improve the way we buy and sell real estate.

Ultimately, this project serves as an example of the potential of data analytics in generating actionable insights and driving business success.

# 7. Next steps
  1. Improve the EDA to bring more interesting insights.
  2. Elaborate more hypotheses to be tested and generate new insights
  3. Try different parameters to define the buy and sell criteria.
  4. Develop Machine Learning Models that can help the company improve its performance.
