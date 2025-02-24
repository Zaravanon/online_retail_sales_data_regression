import streamlit as st
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import pickle
import kagglehub
import os

#Title
st.markdown("""
    <h1 style='text-align: center; font-size: 36px; font-weight: bold;'>US RETAIL PROFIT ANALYZER</h1>
    <h3 style='text-align: center; font-size: 20px; font-weight: normal;'>AI-Powered Forecasting Based on Key Parameters</h3>
""", unsafe_allow_html=True)

#Button Markdown

st.markdown("""
    <style>
        .stButton>button {
            background-color: #ff4b4b;  /* Red-Pink like Streamlit slider */
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff6666; /* Lighter red-pink on hover */
        }
    </style>
""", unsafe_allow_html=True)
    

# Sales input
sales = st.number_input("Sales")

# Quantity input (0 to 1 range)
quantity = st.number_input("Quantity", min_value=0, max_value=30, step=1)

# Discount input (0 to 1 range)
discount = st.number_input("Discount(in percentage)", min_value=0, max_value=100, step=1)

# Region_West dropdown
region_west = st.selectbox("Region_West", ["Yes", "No"])


# Sub-category dropdown
sub_category = st.selectbox("Sub-Category", [
    "Bookcases", "Chairs", "Labels", "Tables", "Storage", "Furnishings", "Art", "Phones", "Binders", 
    "Appliances", "Paper", "Accessories", "Envelopes", "Fasteners", "Supplies", "Machines", "Copiers"
])


#dataf = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\All Project Details\Retail_Sales_Data_Project_Files\Sample - Superstore.csv', encoding='latin1')
path = kagglehub.dataset_download("vivek468/superstore-dataset-final")

# Get the first CSV file from the downloaded path
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

# Load the dataset with the correct encoding
dataf = pd.read_csv(os.path.join(path, csv_files[0]), encoding="ISO-8859-1")  # Use 'latin1' if needed

# Create a dictionary to map states to cities
state_city_map = dataf.groupby('State')['City'].unique().to_dict()

# Get unique states
states = list(state_city_map.keys())

# Create state dropdown
states = st.selectbox("Select a State", states, index=states.index("California") if "California" in states else 0)  # Default to California or the first state

# Filter cities based on selected state
cities = state_city_map[states].tolist()

# Create city dropdown
city = st.selectbox("Select a City", cities, index=cities.index("Los Angeles") if "Los Angeles" in cities else 0) #Default to Los Angeles or first city.

ship_day = st.slider("Ship Day", min_value=1, max_value=30, value=1)
order_day = st.slider("Order Day", min_value=1, max_value=30, value=1)
ship_month = st.slider("Ship Month", min_value=1, max_value=12, value=1)


# Initialize session state for both dropdowns
if "category_tech" not in st.session_state:
    st.session_state.category_tech = "No"
if "category_office" not in st.session_state:
    st.session_state.category_office = "No"

# Function to toggle values
def toggle_dropdown(changed):
    if changed == "category_tech" and st.session_state.category_tech == "Yes":
        st.session_state.category_office = "No"  # Ensure both aren't Yes
    elif changed == "category_office" and st.session_state.category_office == "Yes":
        st.session_state.category_tech = "No"  # Ensure both aren't Yes

# Dropdowns with callbacks
category_tech = st.selectbox("Category - Technology", ["Yes", "No"], 
                             index=0 if st.session_state.category_tech == "Yes" else 1, 
                             key="category_tech", 
                             on_change=toggle_dropdown, 
                             args=("category_tech",))

category_office = st.selectbox("Category - Office Supplies", ["Yes", "No"], 
                               index=0 if st.session_state.category_office == "Yes" else 1, 
                               key="category_office", 
                               on_change=toggle_dropdown, 
                               args=("category_office",))


button_comment = """# Submit button
if st.button("Predict"):
    st.success("Form submitted successfully!")"""


#Transformations 


Sales = np.log(sales + 1)
#st.write("Sales:",Sales)



Quantity = quantity
#st.write("Quantity:",Quantity)


comment = """
safe_quantity = np.where(Quantity == 0, 1, Quantity)

# Calculate unit sales price
unit_sales_tr1 = np.round(Sales / safe_quantity, 2)  # Ensures proper rounding

# Apply log transformation safely
unit_sales_tr2 = np.log1p(unit_sales_tr1)  # log(1 + x) to handle zero values properly




Q_transformer = QuantileTransformer()
unit_sales_price = Q_transformer.fit_transform(np.array(unit_sales_tr2).reshape(-1, 1))
st.write("unit_sales_price:",unit_sales_price)
"""




# Ensure historical Quantity values are never zero for fitting
safe_quantity_data = np.where(dataf['Quantity'] == 0, 1, dataf['Quantity'])

# Compute historical unit sales prices
unit_sales_data = np.log1p(dataf['Sales'] / safe_quantity_data).values.reshape(-1, 1)

# Fit QuantileTransformer on historical data
Q_transformer = QuantileTransformer()
Q_transformer.fit(unit_sales_data)

# ✅ Use the Streamlit input variable correctly
safe_quantity_input = max(1, Quantity)  # Ensure it's at least 1

# Compute unit sales price for the user input
unit_sales_tr1_input = np.round(Sales / safe_quantity_input, 2)  # Using input values
unit_sales_tr2_input = np.log1p(unit_sales_tr1_input)  # Apply log transformation

# Transform using fitted QuantileTransformer
unit_sales_price = Q_transformer.transform([[unit_sales_tr2_input]])

#st.write("unit_sales_price:", unit_sales_price[0][0])



#dataf = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\All Project Details\Retail_Sales_Data_Project_Files\Sample - Superstore.csv', encoding='latin1')

sub_cat_target_mean  = dataf[dataf['Sub-Category'] == sub_category]['Profit'].mean().round(2)

sub_cat_target_mean_disc = sub_cat_target_mean.astype(int)


def sub_cat_feat_fun(sub_category):
    if sub_category in [4, 16]:
        return 1
    elif sub_category in [2, 8, 15]:
        return 2
    elif sub_category in [6]:
        return 4
    else:
        return 3

sub_cat_feat = sub_cat_feat_fun(sub_cat_target_mean_disc)
#st.write("sub_cat_feat: ",sub_cat_feat)




City_tar_mean = dataf[dataf['City'] == city]['Profit'].mean().round(2)
#st.write("City Target Mean: ",City_tar_mean)


State_tar_mean = dataf[dataf['State'] == states]['Profit'].mean().round(2)
#st.write("State Target Mean: ",State_tar_mean)


city_state = City_tar_mean*State_tar_mean

dataf['State'] = dataf.groupby('State')['Profit'].transform('mean').round(2) #Target encoding in dataf
dataf['City'] = dataf.groupby('City')['Profit'].transform('mean').round(2) #Target encoding in dataf


dataf['city_state'] = dataf['City']*dataf['State']

# Calculate frequency of each unique value in 'city_state'
city_state_counts = dataf['city_state'].value_counts()

# Map frequencies to the 'city_state' column
dataf['city_state_encoded'] = dataf['city_state'].map(city_state_counts)

# Extract the frequency of the scalar value
city_state_encoded = city_state_counts.get(city_state, 0)  # Returns 0 if scalar is not in the DataFrame

#st.write("city_state_encoded:", city_state_encoded)

Ship_day = ship_day
#st.write("Ship_day:", Ship_day)



Order_day = order_day
#st.write("Order_day:", Order_day)

Ship_month = ship_month
#st.write("Ship_month:", Ship_month)

if category_tech == "Yes":
    Category_Technology = 1
else:
    Category_Technology = 0

#st.write("Category_Technology:", Category_Technology)


if category_office == "Yes":
    Category_Office_Supplies = 1
else:
    Category_Office_Supplies = 0

#st.write("Category_Office_Supplies:", Category_Office_Supplies)



Discount = discount/100
#st.write("Discount:", Discount)





#Loading pickle
with open(r"online_sales_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the trained Quantile Transformer used for profit target variable
with open(r"quantile_transformer_profit.pkl", "rb") as file:
    qt_pro = pickle.load(file)  # Load the saved QuantileTransformer



loss_img = r"loss_image.webp"  # Replace with the actual file path
no_profit_img = r"nuetral_image.webp"
more_profit_img = r"profit_image.webp"


if st.button("Predict"):
    # Prepare input features in the correct order
    input_features = np.array([[  
        Discount,  
        Sales,  
        unit_sales_price[0][0],  
        sub_cat_feat,  
        city_state_encoded,  
        Ship_day,  
        Order_day,  
        Ship_month,  
        Category_Technology,  
        Category_Office_Supplies,  
        Quantity  
    ]])

    # Ensure input shape is correct for the model
    prediction = model.predict(input_features)  

    # Reshape if necessary before inverse transform
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)

    # Inverse transform using qt_pro (QuantileTransformer for Profit)
    y_pred_original_tr = qt_pro.inverse_transform(prediction)
    y_pred_original = round(y_pred_original_tr[0][0],2)
    # Display the result
    #st.write("Predicted Profit:", round(y_pred_original[0][0], 2))  

    if y_pred_original > 30:
        st.image(more_profit_img, caption="Good Profit", use_container_width =True)
        st.success("So much cash, need a money tub!")
        st.success(f"✅ Profit Generated: ${"{:.2f}".format(y_pred_original)}")

    elif y_pred_original > 0 and y_pred_original <= 30:
        st.image(no_profit_img, caption="Not a good profit", use_container_width =True)
        st.warning("Steady profits, but no Ferrari yet!")
        st.info(f"⚖️ Not a good Profit, No Loss – Generated: ${"{:.2f}".format(y_pred_original)}")

    else:
        st.image(loss_img, caption="Financial Loss", use_container_width =True)
        st.error("Our profits went on a world tour!")
        st.error(f"📉 Loss Incurred: ${abs(y_pred_original):.2f}")

# Initialize session state for button click
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

# Function to update button state
def on_predict():
    st.session_state.predict_clicked = True

# Title
st.markdown("""
    <h1 style='text-align: center; font-size: 36px; font-weight: bold;'>US RETAIL PROFIT ANALYZER</h1>
    <h3 style='text-align: center; font-size: 20px; font-weight: normal;'>AI-Powered Forecasting Based on Key Parameters</h3>
""", unsafe_allow_html=True)

# Styling for Predict button
button_style = """
    <style>
        .stButton>button {
            background-color: #ff4b4b; /* Default Red-Pink */
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff6666; /* Lighter red-pink on hover */
        }
"""

# If button is clicked, update styling dynamically
if st.session_state.predict_clicked:
    button_style += """
        .stButton>button {
            background-color: #4CAF50 !important; /* Green to indicate action */
        }
    """

button_style += "</style>"
st.markdown(button_style, unsafe_allow_html=True)


st.success("Prediction Successful!")  # Or replace this with your actual prediction logic
    
