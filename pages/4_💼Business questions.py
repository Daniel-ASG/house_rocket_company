# ==========================================================
#                   Import libraries
# ==========================================================
import plotly.express    as px
import streamlit         as st
import pandas            as pd
import geopandas

from PIL import Image

# ==========================================================
#                       Functions
# ==========================================================

@st.cache_data
def create_features(df_raw):
    df_raw['basement'] = df_raw.sqft_basement.apply(lambda x: 'No' if x==0 else 'Yes')
    df_raw['standard'] = df_raw.apply(lambda row: 'low_standard' if (row.price<df_raw.price.quantile(0.75))&(row.condition!=5)&(row.grade<=10 ) else                                                             'high_standard', 
                                      axis=1)
    df_raw['house_age'] = df_raw.yr_built.apply(lambda x: 'new_house' if x > 2014 else 'old_house')
    df_raw['renovated'] = df_raw.yr_renovated.apply(lambda x: 'No' if x==0 else 'Yes')
    dormitory_types = ['no_bedrooms', 'studio', 'apartament', 'house']
    df_raw['dormitory_type'] = [dormitory_types[min(x, 3)] for x in df_raw.bedrooms]
    df_raw['waterfront'] = df_raw.waterfront.apply(lambda x: 'No' if x==0 else 'Yes')
    df_raw['price_per_sqft'] = df_raw.price / df_raw.sqft_living
    return df_raw

    
@st.cache_data
def categorize_features(df_raw):
    cols_to_categorize = ['id', 'waterfront', 'view', 'zipcode', 'season', 'basement', 'standard', 'house_age', 'renovated', 'dormitory_type']
    df_raw[cols_to_categorize] = df_raw[cols_to_categorize].apply(pd.Categorical)
    return df_raw


@st.cache_data
def insert_city_names(df_raw, _king_county):
    # create a GeoDataFrame with 'id' and the coordinates ('long', 'lat')
    geometry = geopandas.points_from_xy(df_raw.long, df_raw.lat)
    geo_df_raw = geopandas.GeoDataFrame(df_raw['id'], geometry=geometry)

    # checking if a point in 'geo_df_raw' is inside a polygon in 'king_county'
    pointInPolys = geopandas.tools.sjoin(geo_df_raw, 
                                         king_county, 
                                         op="within", 
                                         how='left')

    # merging 'pointInPolys' with the original 'df_raw'
    df_raw = pd.merge(df_raw, pointInPolys[['id', 'NAME']], how='left', on=['id', 'id'])
    df_raw.rename(columns={'NAME':'city'}, inplace=True)
    allowed_vals = ['Seattle', 'Northshore', 'Lake Washington', 'Federal Way', 'Highline', 'Tahoma', 'Bellevue','Riverview', 'Auburn', 'Mercer Island', 'Kent', 'Issaquah', 'Renton', 'Vashon Island', 'Snoqualmie Valley','Shoreline', 'Enumclaw', 'Tukwila', 'Fife', 'Skykomish']
    df_raw.loc[~df_raw["city"].isin(allowed_vals), "city"] = "undefined"
    df_raw['city'] = pd.Categorical(df_raw.city)
    
    return df_raw

    
@st.cache_data
def extract_features(df_raw):
    df_raw = create_features(df_raw)
    df_raw = categorize_features(df_raw)
    df_raw = insert_city_names(df_raw, king_county)

    # Reordering cols
    cols_order = ['id', 'date', 'season', 'year_sold', 'year_month', 'year_week', 'yr_built', 'house_age', 'yr_renovated', 'renovated',
                     'price', 'price_per_sqft', 'sqft_living', 'sqft_above', 'sqft_basement', 'basement', 'sqft_lot', 'sqft_living15', 'sqft_lot15',
                     'bedrooms', 'dormitory_type', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'standard',
                     'zipcode', 'lat', 'long', 'city']
    
    # Drop duplicates and sort by date
    df = (df_raw[cols_order]
          .drop_duplicates()
          .sort_values(by='date')
          .drop_duplicates(subset=['id'], keep='last'))
    
    return df


@st.cache_data
def feature_engineering(df_raw):
    # Some data transformation and feature engineering
    df_raw['date'] = df_raw['date'].str.replace('T000000', '')
    df_raw.date = pd.to_datetime(df_raw.date)
    df_raw['year_sold'] = df_raw.date.dt.year
    df_raw['year_month'] = df_raw.date.dt.strftime('%Y-%m')
    df_raw['year_week'] = df_raw.date.dt.strftime('%Y-%U')
    df_raw['date'] = df_raw['date'].dt.strftime('%Y-%m-%d')
    df_raw['season'] = ['Spring' if (x <= '2014-06-20') or (x >= '2015-03-20') else 
                        ('Summer' if (x >= '2014-06-21') and (x <= '2014-09-21') else 
                        ('Fall' if (x >= '2014-09-22') and (x <= '2014-12-20') else 
                        'Winter')) for x in df_raw.date]

    # correcting some errors
    # https://blue.kingcounty.com/Assessor/eRealProperty/default.aspx
    # to make some corrections each property with inconsistent data was searched on the site above to see what changes needed to be made
    replacements = {
        2569500210: {'bedrooms': 3},
        6306400140: {'bedrooms': 5, 'bathrooms': 4.5},
        3918400017: {'bedrooms': 3, 'bathrooms': 2.25},
        2954400190: {'bedrooms': 4, 'bathrooms': 4},
        2310060040: {'bedrooms': 4, 'bathrooms': 2.5},
        7849202190: {'bedrooms': 3, 'bathrooms': 1.5},
        9543000205: {'bedrooms': 2, 'bathrooms': 1},
        1222029077: {'bedrooms': 1, 'bathrooms': 1.5}
    }
    df_raw = df_raw.replace({'id': replacements})
    
    # what could not be changed was removed
    index_to_drop = [index for index in df_raw.index if df_raw.loc[index, 'bedrooms'] == 33 or 
                                                        df_raw.loc[index, 'bathrooms'] == 0 or 
                                                        df_raw.loc[index, 'bedrooms'] == 0]
    df_raw.drop(index_to_drop, inplace=True)
            
    return extract_features(df_raw)


@st.cache_data
def create_business_question_df(df):
    df_aux = df[['price', 'zipcode']].groupby('zipcode').median().reset_index()
    df_aux.columns = ['zipcode', 'median_price_region']
    
    # df_bq -> business questions df
    df_bq = pd.merge(df.copy(), df_aux, on='zipcode', how='inner')
    df_bq['buy'] = ['Yes' if (row.price<=row.median_price_region)&(row.condition>=3)&(row.grade>=5) else 
                    'No' for index, row in df_bq.iterrows()]
    columns = ['id', 'buy', 'date', 'season', 'year_sold', 'year_month', 'year_week', 'yr_built', 
               'house_age', 'yr_renovated', 'renovated', 'price', 'median_price_region',
               'price_per_sqft', 'sqft_living', 'sqft_above', 'sqft_basement', 
               'basement', 'sqft_lot', 'sqft_living15', 'sqft_lot15',
               'bedrooms', 'dormitory_type', 'bathrooms', 'floors',
               'waterfront', 'view', 'condition', 'grade',
               'standard','zipcode', 'lat', 'long','city']
    df_bq = df_bq[columns]
    st.dataframe(df_bq)
    
    return df_bq




def map_cities(data):
    fig = px.scatter_mapbox(data,
                            title="<b>median_price_region</b>",
                            lat='lat',
                            lon='long',
                            size='price',
                            color='median_price_region',
                            hover_data=["city"],
                            color_continuous_scale=px.colors.sequential.Cividis,
                            size_max=15,
                            zoom=8.5)

    fig.update_layout(mapbox_style='carto-positron',
                      height=600,
                      margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      title_x=0.05,
                      title_y=0.9,
                      title_font_size=50,
                      title_font_color='black',
                      title_font_family='Arial'
                     )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    
def process_sell_df(df_bq):
    df_sell = df_bq.query('buy == "Yes"')
    df_sell['season_zipcode'] = df_sell.season.astype(str)+'_'+df_sell.zipcode.astype(str)
    df_aux = df_sell.groupby(['season_zipcode']).price.median().reset_index().rename(columns={'price': 'median_price_season_zip'})
    df_sell = pd.merge(df_sell, df_aux, on='season_zipcode', how='inner')

    df_sell['sell_price'] = [1.1*(row.price) if row.price>=row.median_price_season_zip else
                             1.3*(row.price) if (row.price<row.median_price_season_zip)&(row.standard=='low_standard') else
                             1.35*(row.price) for index, row in df_sell.iterrows()]

    df_sell['profit'] = df_sell.sell_price - df_sell.price
    df_sell['profit_percentage'] = (df_sell.profit / df_sell.price) * 100
    df_sell['color'] = ['within the criteria' if (condition > 3 and grade > 8) else 
                        'out of the criteria' for condition, grade in zip(df_sell['condition'], df_sell['grade'])]
    st.dataframe(df_sell[["id", "price", "buy", "sell_price", "profit", "profit_percentage", "city"]])

    frames = pd.DataFrame()
    for i in range(3,6):
        for j in range (5,9):
            df_criteria = df_sell.query('condition>=@i & grade>=@j')
            df_aux = df_criteria[['buy', 'price', 'sell_price', 'profit']].groupby('buy').sum().reset_index()
            df_temp = pd.DataFrame([df_aux.price[0], 
                                    df_aux.sell_price[0], 
                                    df_aux.profit[0], 
                                    df_aux.profit[0] / df_aux.price[0] * 100, 
                                    df_aux.profit[0] / len(df_criteria), 
                                    len(df_criteria)],
                                   index=['total_buy_price', 'total_sell_price', 'profit', 'profit_percentage', 'average_profit', 'n_properties'],
                                 columns =[f'condition>={i} / grade>={j}'])
            frames = pd.concat([frames, df_temp], axis=1)

    frames = frames.T.reset_index()
    frames.rename(columns={'index':'criteria'}, inplace=True)
    frames['% properties in the database'] = frames['n_properties']/len(data)*100
    frames = frames.sort_values(by=['profit'], ascending=False).round(2)
    st.dataframe(frames)
    return frames




def scatter_price_comparison(frames):
    fig = px.scatter(frames, x="profit_percentage", y="average_profit", 
                     color="criteria", 
                     color_discrete_sequence=px.colors.qualitative.Light24,
                     size='profit', 
                     hover_data=['% properties in the database'])
    st.plotly_chart(fig, use_container_width=True)
    
# ----------------------------- Start of the logical code structure -----------------------------

st.set_page_config(page_title='Business questions',
                   page_icon='💼',
                   layout='wide'
                  )

# -------------------
# Import Datasets
# -------------------
kc_house_data = pd.read_csv("dataset/kc_house_data.csv")
king_county = geopandas.read_file('dataset/King_County_Shape_File/School_Districts_in_King_County___schdst_area.shp')

# -------------------
# Feature engineering
# -------------------
data = feature_engineering(kc_house_data)

# ==========================================================
#                       Sidebar
# ==========================================================
image = Image.open('images/logo.png')
st.sidebar.image(image, width=250)

st.sidebar.markdown('''___''')
st.sidebar.markdown('##### Powered by Comunidade DS')
st.sidebar.markdown('##### Data Analyst: Daniel Gomes')



# ==========================================================
#                  Layout in Streamlit
# ==========================================================

st.title('Business questions')
st.markdown(
    '''
    
A real estate transaction is a unique negotiation, as it’s impossible to replicate the exact characteristics in two different units. They will be physically arranged on different floors, in different locations, and have varying degrees of solar exposure, among other factors.

Such characteristics will affect prices and the analysis of available data on properties in a region will favor the company that can choose this tool to conduct its business and maximize its profits through the combined use of geographic data, physical characteristics of properties and even data relating to the population of the area.

Ultimately, the goal is to balance the interests of both, buyers and sellers, to ensure a sustainable market that satisfies all parties. This balance can only be achieved by leveraging data science to eliminate human biases and make the transaction as fair as possible.

Based on the exploratory analysis, the CEO should buy the properties:
    
    - in winter, mainly in February;
    - properties close to the water but without a waterview;

Then, if possible, a renovation could be carried out to sell the property by the end of spring, preferably in April. In this reform the recommendation would be to:

    - build a new floor that would allow the property to:
         * get a waterview;
         * improve the level of the "view" category of the property;
         * increase the sqft_living;
         * increase the amount of floors;
         * increase the amount of rooms.

On the other hand, **House Rocket** management raised some initial questions to the team of data scientists to better understand what is available on the market:

1. [Which houses should the CEO of **House Rocket** buy and at what purchase price?](#1)
    - purchase criteria:
        * group by **zipcode**
        * **price**     < median
        * **condition** ≥ 3
        * **grade**     ≥ 5
        
        
2. [Once the house is in the possession of the company, what would be the price of the sale?](#2)
    - selling criteria:
        - group by **zipcode, season**
        - **price** > median
            - purchase price + 10%
        - **price** < median
            - purchase price + 30%
            - purchase price + 35% for **high_standard**


    ''')

# 3. [Extra questions](#3)

st.subheader('1. Which houses should the CEO of **House Rocket** buy and at what purchase price?',
             anchor='1')
df_bq = create_business_question_df(data)
map_cities(df_bq)




st.subheader('2. Once the house is in the possession of the company, what would be the price of the sale?',
             anchor='2')
frames = process_sell_df(df_bq)
scatter_price_comparison(frames)
st.markdown('''
#### With this result, the CEO is able to decide what his priority will be based on the criteria mentioned above. 

- **Obtaining the highest possible profit while having to invest a considerable amount of money ("condition>=3 / grade>=5" or "condition>=3 / grade>=6")**
    * These two solutions are very similar. Substantial investment is required due to the significant volume of acquisitions that must be made. Nevertheless, to achieve the utmost profitability while adhering to the minimum property quality standard, these conditions **better align** with market preferences.
''')
st.markdown('''
- **Make a more moderate investment and obtain an average profit per property optimized (condition>=5 / grade>=8)** 
    * This option offers the opportunity to attain significant profitability while upholding pre-established quality standards, with an emphasis on a limited selection of properties that can yield favorable results through the marketing of a reduced quantity. Targeted towards **high-end clientele**, it epitomizes the pinnacle of House Rocket's portfolio, ensuring the highest returns per property.
''')
st.dataframe(frames.iloc[[0,1,11]].set_index('criteria'))

# melhorar a análise para mostrar quais foram as casas compradas com preços baixos na baixa estação e quais devem ser vendidas pelo mais alto valor possivel na alta estação

# no fututo, fazer a avaliação em função da data da compra (MUITO IMPORTANTE)