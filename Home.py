# ==========================================================
#                   Import libraries
# ==========================================================
import streamlit as st

from PIL import Image

# ==========================================================
#                       Functions
# ==========================================================

# ----------- Start of the logical code structure ----------

st.set_page_config(page_title='Home',
                   page_icon='🏠',
                   layout='wide'
                  )


# ==========================================================
#                       Sidebar
# ==========================================================
image = Image.open('images/logo.png')
st.sidebar.image(image, width=250)

st.sidebar.markdown('# House Rocket Company')

st.sidebar.markdown('''___''')
st.sidebar.markdown('##### Powered by Comunidade DS')
st.sidebar.markdown('##### Data Analyst: Daniel Gomes')


# ==========================================================
#                  Layout in Streamlit
# ==========================================================
image = Image.open('images/pexels-josh-fields-3964406.jpg')
st.image(image, 
         caption='Seattle, WA, United States. Photo by Josh Fields: https://www.pexels.com/photo/aerial-view-of-city-buildings-3964406/'
        )

st.write("# House Rocket Company Dashboard")
st.markdown(
    '''
    ### Welcome to House Rocket Data Analysis! 
    ### This dashboard was designed to help the company's CEO make decisions regarding the real estate purchase and sale in the King Count region.
    ### How to use this Dashboard?
    - Overview:
        - General behavioral metrics.
        - Geolocation distribution.
        
    - EDA:
        - Exploratory data analysis.
        
    - Hypotheses:
        - Business team hypotheses.
        
    - Business questions:
        - Solution of the business problem presented by the CEO.
        
    ### Ask for help
    - Data Science Team on Discord
        - @daniel_asg
    - For more information, please visit the [project page on GitHub](https://github.com/Daniel-ASG/dishy_company/tree/main). Thanks for your visit.
    
    ''')