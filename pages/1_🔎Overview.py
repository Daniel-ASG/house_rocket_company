# ==========================================================
#                   Import libraries
# ==========================================================
import plotly.express    as px
import streamlit         as st
import pandas            as pd
import numpy             as np
import os
os.environ["USE_PYGEOS"] = "0"
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



def map_cities(data):
    fig = px.scatter_mapbox(data,
                            title="<b>Cities Distribution</b>",
                            lat='lat',
                            lon='long',
                            size='price',
                            color='city',
                            hover_data=["zipcode"],
                            color_discrete_sequence=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'],
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



# ----------------------------- Start of the logical code structure -----------------------------

st.set_page_config(page_title='Overview',
                   page_icon='🔎',
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

st.sidebar.markdown('# Project overview')

f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
f_zipcode    = st.sidebar.multiselect('Enter zipcodes', data.zipcode.unique())
f_city       = st.sidebar.multiselect('Enter cities', data.city.unique())

data_overview = data.copy()
if f_zipcode:
    data_overview = data_overview.loc[data_overview.zipcode.isin(f_zipcode)]
if f_attributes:
    data_overview = data_overview[f_attributes]
if f_city:
    data_overview = data_overview.loc[data_overview.city.isin(f_city)]

st.sidebar.markdown('''___''')
st.sidebar.markdown('##### Powered by Comunidade DS')
st.sidebar.markdown('##### Data Analyst: Daniel Gomes')




# ==========================================================
#                  Layout in Streamlit
# ==========================================================

st.title('Data overview')
st.markdown(
    '''
## Business Question
    
House Rocket is a (fictitious) digital platform whose business model is to buy and sell real estate using technology.

The company wants to find the best business opportunities in the real estate market. The CEO of House Rocket would like to maximize the company's revenue by finding good business opportunities.

The main strategy is to __buy GOOD HOUSES__ in great locations at __LOW PRICES__ and then __resell them at HIGHER PRICES__. The greater the difference between buying and selling, the greater the company's profit and therefore its revenue.

However, houses have many attributes that make them more or less attractive to buyers and sellers. Location and the time of year the property is traded can also influence prices.
    
## Business Understanding

1. Which houses should the CEO of House Rocket buy and for what purchase price?
2. Once the houses are in the company's possession, when would be the best time to sell them and what would be the sale price?
3. Should House Rocket renovate to increase the sale price? What would be the suggested changes? What is the increase in price for each renovation option?

## Set of hypotheses

* Are houses with garages more expensive? Why is that?
* Are houses with many bedrooms more expensive? Why is that? From how many rooms does the price increase? What is the price increase for each room added?
* Are the most expensive houses in the center? Which region? Is there anything in the area that correlates with the house's sale price? Shopping centers? Mountains? Famous people?
    ''')


st.markdown(
    '''
## Dataset overview
The dataset that represents the context of the problem addressed is available on the Kaggle platform.
This is the link: https://www.kaggle.com/harlfoxem/housesalesprediction

This dataset contains houses sold between May 2014 and May 2015.
    ''')
st.dataframe(data_overview)
st.markdown(
    '''
| Original variables dictionary |                                                                                                                                                                                                      |
|:------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|            id            | Unique ID for each home sold                                                                                                                                                                         |
|           date           | Date of the home sale                                                                                                                                                                                |
|           price          | Price of each home sold                                                                                                                                                                              |
|         bedrooms         | Number of bedroom                                                                                                                                                                                    |
|         bathrooms        | Number of bathrooms, where .5 accounts for a room with a toilet but no shower                                                                                                                        |
|        sqft_living       | Square footage of the apartments interior living space (sqft_living = sqft_above + sqft_basement)                                                                                                                                              |
|         sqft_lot         | Square footage of the land space                                                                                                                                                                     |
|          floors          | Number of floors                                                                                                                                                                                     |
|        waterfront        | A dummy variable for whether the apartment was overlooking the waterfront or not                                                                                                                     |
|           view           | An index from 0 to 4 of how good the view of the property was                                                                                                                                        |
|         condition        | An index from 1 to 5 on the condition of the apartment,                                                                                                                                              |
|           grade          | An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design. |
|        sqft_above        | The square footage of the interior housing space that is above ground level                                                                                                                          |
|       sqft_basement      | The square footage of the interior housing space that is below ground level                                                                                                                          |
|         yr_built         | The year the house was initially built                                                                                                                                                               |
|       yr_renovated       | The year of the house’s last renovation                                                                                                                                                              |
|          zipcode         | What zipcode area the house is in                                                                                                                                                                    |
|            lat           | Lattitude                                                                                                                                                                                            |
|           long           | Longitude                                                                                                                                                                                            |
|       sqft_living15      | The square footage of interior housing living space for the nearest 15 neighbors                                                                                                                     |
|        sqft_lot15        | The square footage of the land lots of the nearest 15 neighbors                                                                                                                                      |

> Source: <https://www.kaggle.com/harlfoxem/housesaleprediction/discussion/207885>
    ''')


st.markdown("## Map overview")
map_cities(data[['lat', 'long', 'price', 'city', "zipcode"]])



st.markdown(
    '''        
### Ask for help
- Data Science Team on Discord
    - @daniel_asg
- For more information, please visit the [project page on GitHub](https://github.com/Daniel-ASG/dishy_company/tree/main). Thanks for your visit.
    
    ''')

