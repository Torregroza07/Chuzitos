import pandas as pd

df = pd.read_csv("restaurante_baq_demanda_500_dias_2025_2026.csv")
df.describe()
df

costs = {
    'Perros_calientes': 6000,
    'Pizza': 10000,
    'Hamburguesa': 17000,
    'Salchipapa': 12000,
    'Chuzo_Desgranado': 9000,
    'Asados': 8000,
    'Coca_cola': 1500,
    'Limonada': 5000,
    'Cerveza': 1500
}

selling_prices = {
    'Perros_calientes': 15000,
    'Pizza': 20000,
    'Hamburguesa': 25000,
    'Salchipapa': 17000,
    'Chuzo_Desgranado': 15000,
    'Asados': 15000,
    'Coca_cola': 6000,
    'Limonada': 10000,
    'Cerveza': 4000
}

# Calculate profit for each product for each day
for product in product_columns:
    df[f'{product}_ganancia'] = (selling_prices[product] - costs[product]) * df[product]

# Calculate total daily profit ('ganancia')
df['ganancia'] = df[[f'{product}_ganancia' for product in product_columns]].sum(axis=1)

df['holiday_name'] = df['holiday_name'].str.replace('Corpus Christi', 'Carnavales')
print("Replaced 'Corpus Christi' with 'Carnavales' in 'holiday_name' column.")

print("New columns for individual product profit and 'ganancia' (total daily profit) have been created.")

df.columns

import matplotlib.pyplot as plt
import seaborn as sns

df['date'] = pd.to_datetime(df['date'])
print("Plotting libraries imported and 'date' column converted to datetime.")

product_columns = ['Perros_calientes', 'Pizza', 'Hamburguesa', 'Salchipapa', 'Chuzo_Desgranado', 'Asados', 'Coca_cola', 'Limonada', 'Cerveza']

plt.figure(figsize=(15, 8))

for column in product_columns:
    sns.lineplot(data=df, x='date', y=column, label=column)

plt.title('Product Demand Over Time')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

correlation_matrix = df[product_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Product Demand')
plt.show()

weekday_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

average_demand_per_weekday = df.groupby('weekday')[product_columns].mean().reset_index()

plt.figure(figsize=(15, 8))

average_demand_per_weekday_melted = average_demand_per_weekday.melt(id_vars='weekday', var_name='Product', value_name='Average Demand')
sns.barplot(data=average_demand_per_weekday_melted, x='weekday', y='Average Demand', hue='Product', palette='tab10')

plt.title('Average Product Demand per Weekday')
plt.xlabel('Weekday')
plt.ylabel('Average Demand')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.df['total_demand'] = df[product_columns].sum(axis=1)
print("New column 'total_demand' created.")show()

holiday_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')['total_demand'].mean().reset_index()
highest_demand_holiday = holiday_demand.loc[holiday_demand['total_demand'].idxmax()]

print("Average total daily demand per holiday (excluding 'Ninguno'):")
print(holiday_demand)
print("\nHoliday with the highest average total demand:")
print(highest_demand_holiday)

specific_products = ['Cerveza', 'Salchipapa']
holiday_product_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')[specific_products].mean().reset_index()

print("Average demand for 'Cerveza' and 'Salchipapa' per holiday (excluding 'Ninguno'):")
print(holiday_product_demand)

overall_average_demand = df[specific_products].mean()
carnavales_demand = holiday_product_demand[holiday_product_demand['holiday_name'] == 'Carnavales'][specific_products].iloc[0]

print("\nOverall Average Demand for Cerveza and Salchipapa:")
print(overall_average_demand)

print("\nAverage Demand for Cerveza and Salchipapa during Carnavales:")
print(carnavales_demand)

print("\nComparison: Carnavales demand vs. Overall Average Demand:")
print(carnavales_demand / overall_average_demand)

holiday_product_demand_melted = holiday_product_demand.melt(id_vars='holiday_name', var_name='Product', value_name='Average Demand')

plt.figure(figsize=(15, 8))

sns.barplot(
    data=holiday_product_demand_melted,
    x='holiday_name',
    y='Average Demand',
    hue='Product',
    palette={'Cerveza': 'skyblue', 'Salchipapa': 'lightcoral'}
)

# Highlight 'Carnavales'
carnavales_data = holiday_product_demand_melted[holiday_product_demand_melted['holiday_name'] == 'Carnavales']
plt.scatter(
    x=carnavales_data['holiday_name'],
    y=carnavales_data['Average Demand'],
    color='red',
    s=100,
    zorder=5,
    label='Carnavales Highlight'
)

plt.title('Average Demand for Cerveza and Salchipapa per Holiday')
plt.xlabel('Holiday Name')
plt.ylabel('Average Demand')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


holiday_product_demand_melted = holiday_product_demand.melt(id_vars='holiday_name', var_name='Product', value_name='Average Demand')

plt.figure(figsize=(15, 8))

sns.barplot(
    data=holiday_product_demand_melted,
    x='holiday_name',
    y='Average Demand',
    hue='Product',
    palette={'Cerveza': 'skyblue', 'Salchipapa': 'lightcoral'}
)

# Highlight 'Carnavales'
carnavales_data = holiday_product_demand_melted[holiday_product_demand_melted['holiday_name'] == 'Carnavales']
plt.scatter(
    x=carnavales_data['holiday_name'],
    y=carnavales_data['Average Demand'],
    color='red',
    s=100,
    zorder=5,
    label='Carnavales Highlight'
)

plt.title('Average Demand for Cerveza and Salchipapa per Holiday')
plt.xlabel('Holiday Name')
plt.ylabel('Average Demand')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


df['holiday_name'] = df['holiday_name'].replace(['Jueves Santo', 'Viernes Santo','Todos los Santos'], 'Semana Santa')
df['holiday_name'] = df['holiday_name'].replace('Asunción', 'Carnavales')
print("Holiday names 'Jueves Santo' and 'Viernes Santo' grouped into 'Semana Santa'.")
print("Holiday name 'Asunción' consolidated into 'Carnavales'.")


holiday_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')['total_demand'].mean().reset_index()
highest_demand_holiday = holiday_demand.loc[holiday_demand['total_demand'].idxmax()]

print("Average total daily demand per holiday (excluding 'Ninguno'):")
print(holiday_demand)
print("\nHoliday with the highest average total demand:")
print(highest_demand_holiday)


specific_products = ['Cerveza', 'Salchipapa']
holiday_product_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')[specific_products].mean().reset_index()

print("Average demand for 'Cerveza' and 'Salchipapa' per holiday (excluding 'Ninguno'):")
print(holiday_product_demand)


overall_average_demand = df[specific_products].mean()
carnavales_demand = holiday_product_demand[holiday_product_demand['holiday_name'] == 'Carnavales'][specific_products].iloc[0]

print("\nOverall Average Demand for Cerveza and Salchipapa:")
print(overall_average_demand)

print("\nAverage Demand for Cerveza and Salchipapa during Carnavales:")
print(carnavales_demand)

print("\nComparison: Carnavales demand vs. Overall Average Demand:")
print(carnavales_demand / overall_average_demand)


holiday_product_demand_melted = holiday_product_demand.melt(id_vars='holiday_name', var_name='Product', value_name='Average Demand')

plt.figure(figsize=(15, 8))

sns.barplot(
    data=holiday_product_demand_melted,
    x='holiday_name',
    y='Average Demand',
    hue='Product',
    palette={'Cerveza': 'skyblue', 'Salchipapa': 'lightcoral'}
)

# Highlight 'Carnavales'
carnavales_data = holiday_product_demand_melted[holiday_product_demand_melted['holiday_name'] == 'Carnavales']
plt.scatter(
    x=carnavales_data['holiday_name'],
    y=carnavales_data['Average Demand'],
    color='red',
    s=100,
    zorder=5,
    label='Carnavales Highlight'
)

plt.title('Average Demand for Cerveza and Salchipapa per Holiday')
plt.xlabel('Holiday Name')
plt.ylabel('Average Demand')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=holiday_demand, x='holiday_name', y='total_demand', palette='viridis')

plt.title('Average Total Daily Demand per Holiday')
plt.xlabel('Holiday Name')
plt.ylabel('Average Total Demand')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 8))
sns.barplot(data=holiday_demand, x='holiday_name', y='total_demand', hue='holiday_name', palette='viridis', legend=False)

plt.title('Average Total Daily Demand per Holiday')
plt.xlabel('Holiday Name')
plt.ylabel('Average Total Demand')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#!pip install streamlit
import streamlit as st
import plotly.express as px
import pandas as pd

# Assume 'df' DataFrame is already loaded and preprocessed (date column is datetime)
# from the previous cells.

product_columns = ['Perros_calientes', 'Pizza', 'Hamburguesa', 'Salchipapa', 'Chuzo_Desgranado', 'Asados', 'Coca_cola', 'Limonada', 'Cerveza']

# Melt the DataFrame to long format for Plotly Express to create multiple lines
df_melted = df.melt(id_vars=['date'], value_vars=product_columns, var_name='Product', value_name='Demand')

# Title for the Streamlit app
st.title("Product Demand Visualization Over Time")

# Create Plotly Multi-line Chart for product demand over time
fig = px.line(df_melted, x='date', y='Demand', color='Product',
              title='Product Demand Over Time (All Specified Products)',
              labels={'date': 'Date', 'Demand': 'Demand Quantity', 'Product': 'Product Type'})

# Update layout for better readability
fig.update_layout(hovermode="x unified") # Shows all y-values on hover for a given x-value

# Display Plot in Streamlit
st.plotly_chart(fig, use_container_width=True)


# Codigo para el dashboard

import streamlit as st
import plotly.express as px
import pandas as pd

# Ensure 'weekday' column is categorical and ordered
weekday_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

# Calculate average demand per weekday for product_columns
average_demand_per_weekday = df.groupby('weekday', observed=False)[product_columns].mean().reset_index()

# Melt the DataFrame to long format for Plotly Express
average_demand_per_weekday_melted = average_demand_per_weekday.melt(id_vars='weekday', var_name='Product', value_name='Average Demand')

# Create Plotly Bar Chart
fig_weekday = px.bar(
    average_demand_per_weekday_melted,
    x='weekday',
    y='Average Demand',
    color='Product',
    title='Average Product Demand per Weekday',
    labels={'weekday': 'Weekday', 'Average Demand': 'Average Demand Quantity', 'Product': 'Product Type'},
    barmode='group'
)

# Update layout for better readability
fig_weekday.update_layout(xaxis_title='Weekday', yaxis_title='Average Demand')

# Display Plot in Streamlit
st.plotly_chart(fig_weekday, width='stretch')


import streamlit as st
import plotly.express as px

holiday_product_demand_melted = holiday_product_demand.melt(id_vars='holiday_name', var_name='Product', value_name='Average Demand')

# Create Plotly Bar Chart
fig_holiday = px.bar(
    holiday_product_demand_melted,
    x='holiday_name',
    y='Average Demand',
    color='Product',
    title='Average Demand for Cerveza and Salchipapa per Holiday',
    labels={'holiday_name': 'Holiday Name', 'Average Demand': 'Average Demand Quantity', 'Product': 'Product Type'},
    barmode='group'
)

# Update layout for better readability and highlight Carnavales (optional, can be done with annotations in Plotly if needed)
fig_holiday.update_layout(xaxis_title='Holiday Name', yaxis_title='Average Demand')

# Display Plot in Streamlit
st.plotly_chart(fig_holiday, width='stretch')


import streamlit as st
import plotly.express as px
import pandas as pd

# Ensure product_columns and costs are defined from previous cells
# product_columns = ['Perros_calientes', 'Pizza', 'Hamburguesa', 'Salchipapa', 'Chuzo_Desgranado', 'Asados', 'Coca_cola', 'Limonada', 'Cerveza']
# costs = { ... }

# Calculate daily cost for each product
cost_columns = []
for product in product_columns:
    df[f'{product}_costo_diario'] = costs[product] * df[product]
    cost_columns.append(f'{product}_costo_diario')

# Melt the DataFrame to long format for Plotly Express box plots
df_costs_melted = df.melt(id_vars=['date'], value_vars=cost_columns, var_name='Product Cost', value_name='Daily Cost')

# Title for the Streamlit app
st.title("Daily Product Costs Distribution")

# Create Plotly Box Plot for product costs
fig_boxplot = px.box(df_costs_melted, x='Product Cost', y='Daily Cost',
                     color='Product Cost',
                     title='Distribution of Daily Costs per Product',
                     labels={'Product Cost': 'Product', 'Daily Cost': 'Daily Cost Value'})

# Update layout for better readability
fig_boxplot.update_layout(xaxis_title='Product', yaxis_title='Daily Cost')

# Display Plot in Streamlit
st.plotly_chart(fig_boxplot, width='stretch')


import streamlit as st
import plotly.express as px
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("restaurante_baq_demanda_500_dias_2025_2026.csv")

# 2. Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# 3. Define product_columns, costs, and selling_prices
product_columns = ['Perros_calientes', 'Pizza', 'Hamburguesa', 'Salchipapa', 'Chuzo_Desgranado', 'Asados', 'Coca_cola', 'Limonada', 'Cerveza']

costs = {
    'Perros_calientes': 6000,
    'Pizza': 10000,
    'Hamburguesa': 17000,
    'Salchipapa': 12000,
    'Chuzo_Desgranado': 9000,
    'Asados': 8000,
    'Coca_cola': 1500,
    'Limonada': 5000,
    'Cerveza': 1500
}

selling_prices = {
    'Perros_calientes': 15000,
    'Pizza': 20000,
    'Hamburguesa': 25000,
    'Salchipapa': 17000,
    'Chuzo_Desgranado': 15000,
    'Asados': 15000,
    'Coca_cola': 6000,
    'Limonada': 10000,
    'Cerveza': 4000
}

# 4. Calculate profit for each product and total daily profit
for product in product_columns:
    df[f'{product}_ganancia'] = (selling_prices[product] - costs[product]) * df[product]
df['ganancia'] = df[[f'{product}_ganancia' for product in product_columns]].sum(axis=1)

# 5. Apply all holiday name adjustments
df['holiday_name'] = df['holiday_name'].str.replace('Corpus Christi', 'Carnavales')
df['holiday_name'] = df['holiday_name'].replace(['Jueves Santo', 'Viernes Santo', 'Todos los Santos'], 'Semana Santa')
df['holiday_name'] = df['holiday_name'].replace('Asunción', 'Carnavales')

# 6. Calculate total_demand
df['total_demand'] = df[product_columns].sum(axis=1)

# 7. Define weekday_order and convert 'weekday' column to categorical
weekday_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

# 8. Calculate average_demand_per_weekday and melt for plotting
average_demand_per_weekday = df.groupby('weekday', observed=False)[product_columns].mean().reset_index()
average_demand_per_weekday_melted = average_demand_per_weekday.melt(id_vars='weekday', var_name='Product', value_name='Average Demand')

# 9. Calculate holiday_demand (excluding 'Ninguno')
holiday_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')['total_demand'].mean().reset_index()

# 10. Define specific_products
specific_products = ['Cerveza', 'Salchipapa']

# 11. Calculate holiday_product_demand and melt for plotting
holiday_product_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')[specific_products].mean().reset_index()
holiday_product_demand_melted = holiday_product_demand.melt(id_vars='holiday_name', var_name='Product', value_name='Average Demand')

# 12. Calculate overall_average_demand (not directly used in plots, but for completeness)
overall_average_demand = df[specific_products].mean()

# 13. Calculate daily costs for each product and melt for box plot
cost_columns = []
for product in product_columns:
    df[f'{product}_costo_diario'] = costs[product] * df[product]
    cost_columns.append(f'{product}_costo_diario') # Corrected: Removed extra '_costo'
df_costs_melted = df.melt(id_vars=['date'], value_vars=cost_columns, var_name='Product Cost', value_name='Daily Cost')

# --- New Data Preparation for Monthly Scatter Plot (2025) ---
df_2025 = df[df['date'].dt.year == 2025].copy()
df_2025['month'] = df_2025['date'].dt.month

product_profit_columns = [f'{p}_ganancia' for p in product_columns]

monthly_aggregated_data_detailed = df_2025.groupby('month')[product_columns + product_profit_columns].sum().reset_index()

monthly_demand_melted = monthly_aggregated_data_detailed.melt(
    id_vars='month',
    value_vars=product_columns,
    var_name='Product',
    value_name='Demand'
)

monthly_profit_melted = monthly_aggregated_data_detailed.melt(
    id_vars='month',
    value_vars=product_profit_columns,
    var_name='Product_Profit_Col',
    value_name='Profit'
)

monthly_profit_melted['Product'] = monthly_profit_melted['Product_Profit_Col'].str.replace('_ganancia', '')

monthly_combined_melted = pd.merge(monthly_demand_melted, monthly_profit_melted[['month', 'Product', 'Profit']], on=['month', 'Product'])

# Streamlit App Title
st.title("Product Demand and Cost Analysis Dashboard")

# --- Plot 1: Monthly Product Demand and Profit Scatter Plot (2025) ---

selected_month = st.slider(
    "Select Month for Product Demand and Profit (2025):",
    min_value=int(monthly_combined_melted['month'].min()),
    max_value=int(monthly_combined_melted['month'].max()),
    value=int(monthly_combined_melted['month'].min()),
    step=1
)

filtered_monthly_data = monthly_combined_melted[monthly_combined_melted['month'] == selected_month]

fig_monthly_scatter = px.scatter(
    filtered_monthly_data,
    x='Product',
    y='Demand',
    size='Profit',
    color='Product',
    title=f'Monthly Product Demand and Profit for Month {selected_month} (2025)',
    labels={'Demand': 'Total Demand', 'Profit': 'Total Profit'},
    color_discrete_sequence=px.colors.qualitative.Plotly,
    hover_data={'Product': True, 'Demand': True, 'Profit': True, 'month': True}
)

fig_monthly_scatter.update_layout(hovermode="x unified")
st.plotly_chart(fig_monthly_scatter, width='stretch')

# --- End Plot 1 ---

# Plot 2: Bar plot of average product demand per weekday
fig_weekday = px.bar(
    average_demand_per_weekday_melted,
    x='weekday',
    y='Average Demand',
    color='Product',
    title='Average Product Demand per Weekday',
    labels={'weekday': 'Weekday', 'Average Demand': 'Average Demand Quantity', 'Product': 'Product Type'},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_weekday.update_layout(xaxis_title='Weekday', yaxis_title='Average Demand')
st.plotly_chart(fig_weekday, width='stretch')

# Plot 3: Bar plot of average demand for specific products per holiday
fig_holiday = px.bar(
    holiday_product_demand_melted,
    x='holiday_name',
    y='Average Demand',
    color='Product',
    title='Average Demand for Cerveza and Salchipapa per Holiday',
    labels={'holiday_name': 'Holiday Name', 'Average Demand': 'Average Demand Quantity', 'Product': 'Product Type'},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_holiday.update_layout(xaxis_title='Holiday Name', yaxis_title='Average Demand')
st.plotly_chart(fig_holiday, width='stretch')

# Plot 4: Box plot of daily product costs
fig_boxplot = px.box(df_costs_melted, x='Product Cost', y='Daily Cost',
                     color='Product Cost',
                     title='Distribution of Daily Costs per Product',
                     labels={'Product Cost': 'Product', 'Daily Cost': 'Daily Cost Value'},
                     color_discrete_sequence=px.colors.qualitative.Plotly)
fig_boxplot.update_layout(xaxis_title='Product', yaxis_title='Daily Cost')
st.plotly_chart(fig_boxplot, width='stretch')


import streamlit as st
import plotly.express as px
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("restaurante_baq_demanda_500_dias_2025_2026.csv")

# 2. Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# 3. Define product_columns, costs, and selling_prices
product_columns = ['Perros_calientes', 'Pizza', 'Hamburguesa', 'Salchipapa', 'Chuzo_Desgranado', 'Asados', 'Coca_cola', 'Limonada', 'Cerveza']

costs = {
    'Perros_calientes': 6000,
    'Pizza': 10000,
    'Hamburguesa': 17000,
    'Salchipapa': 12000,
    'Chuzo_Desgranado': 9000,
    'Asados': 8000,
    'Coca_cola': 1500,
    'Limonada': 5000,
    'Cerveza': 1500
}

selling_prices = {
    'Perros_calientes': 15000,
    'Pizza': 20000,
    'Hamburguesa': 25000,
    'Salchipapa': 17000,
    'Chuzo_Desgranado': 15000,
    'Asados': 15000,
    'Coca_cola': 6000,
    'Limonada': 10000,
    'Cerveza': 4000
}

# 4. Calculate profit for each product and total daily profit
for product in product_columns:
    df[f'{product}_ganancia'] = (selling_prices[product] - costs[product]) * df[product]
df['ganancia'] = df[[f'{product}_ganancia' for product in product_columns]].sum(axis=1)

# 5. Apply all holiday name adjustments
df['holiday_name'] = df['holiday_name'].str.replace('Corpus Christi', 'Carnavales')
df['holiday_name'] = df['holiday_name'].replace(['Jueves Santo', 'Viernes Santo', 'Todos los Santos'], 'Semana Santa')
df['holiday_name'] = df['holiday_name'].replace('Asunción', 'Carnavales')

# 6. Calculate total_demand
df['total_demand'] = df[product_columns].sum(axis=1)

# 7. Define weekday_order and convert 'weekday' column to categorical
weekday_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

# 8. Calculate average_demand_per_weekday and melt for plotting
average_demand_per_weekday = df.groupby('weekday', observed=False)[product_columns].mean().reset_index()
average_demand_per_weekday_melted = average_demand_per_weekday.melt(id_vars='weekday', var_name='Product', value_name='Average Demand')

# 9. Calculate holiday_demand (excluding 'Ninguno')
holiday_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')['total_demand'].mean().reset_index()

# 10. Define specific_products
specific_products = ['Cerveza', 'Salchipapa']

# 11. Calculate holiday_product_demand and melt for plotting
holiday_product_demand = df[df['holiday_name'] != 'Ninguno'].groupby('holiday_name')[specific_products].mean().reset_index()
holiday_product_demand_melted = holiday_product_demand.melt(id_vars='holiday_name', var_name='Product', value_name='Average Demand')

# 12. Calculate overall_average_demand (not directly used in plots, but for completeness)
overall_average_demand = df[specific_products].mean()

# 13. Calculate daily costs for each product and melt for box plot
cost_columns = []
for product in product_columns:
    df[f'{product}_costo_diario'] = costs[product] * df[product]
    cost_columns.append(f'{product}_costo_diario') # Corrected: Removed extra '_costo'
df_costs_melted = df.melt(id_vars=['date'], value_vars=cost_columns, var_name='Product Cost', value_name='Daily Cost')

# --- New Data Preparation for Monthly Scatter Plot (2025) ---
df_2025 = df[df['date'].dt.year == 2025].copy()
df_2025['month'] = df_2025['date'].dt.month

product_profit_columns = [f'{p}_ganancia' for p in product_columns]

monthly_aggregated_data_detailed = df_2025.groupby('month')[product_columns + product_profit_columns].sum().reset_index()

monthly_demand_melted = monthly_aggregated_data_detailed.melt(
    id_vars='month',
    value_vars=product_columns,
    var_name='Product',
    value_name='Demand'
)

monthly_profit_melted = monthly_aggregated_data_detailed.melt(
    id_vars='month',
    value_vars=product_profit_columns,
    var_name='Product_Profit_Col',
    value_name='Profit'
)

monthly_profit_melted['Product'] = monthly_profit_melted['Product_Profit_Col'].str.replace('_ganancia', '')

monthly_combined_melted = pd.merge(monthly_demand_melted, monthly_profit_melted[['month', 'Product', 'Profit']], on=['month', 'Product'])

# Streamlit App Title
st.title("Product Demand and Cost Analysis Dashboard")

# --- Plot 1: Monthly Product Demand and Profit Scatter Plot (2025) ---

selected_month = st.slider(
    "Select Month for Product Demand and Profit (2025):",
    min_value=int(monthly_combined_melted['month'].min()),
    max_value=int(monthly_combined_melted['month'].max()),
    value=int(monthly_combined_melted['month'].min()),
    step=1
)

filtered_monthly_data = monthly_combined_melted[monthly_combined_melted['month'] == selected_month]

fig_monthly_scatter = px.scatter(
    filtered_monthly_data,
    x='Product',
    y='Demand',
    size='Profit',
    color='Product',
    title=f'Monthly Product Demand and Profit for Month {selected_month} (2025)',
    labels={'Demand': 'Total Demand', 'Profit': 'Total Profit'},
    color_discrete_sequence=px.colors.qualitative.Plotly,
    hover_data={'Product': True, 'Demand': True, 'Profit': True, 'month': True}
)

fig_monthly_scatter.update_layout(hovermode="x unified")
st.plotly_chart(fig_monthly_scatter, width='stretch')

# --- End Plot 1 ---

# Plot 2: Bar plot of average product demand per weekday
fig_weekday = px.bar(
    average_demand_per_weekday_melted,
    x='weekday',
    y='Average Demand',
    color='Product',
    title='Average Product Demand per Weekday',
    labels={'weekday': 'Weekday', 'Average Demand': 'Average Demand Quantity', 'Product': 'Product Type'},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_weekday.update_layout(xaxis_title='Weekday', yaxis_title='Average Demand')
st.plotly_chart(fig_weekday, width='stretch')

# Plot 3: Bar plot of average demand for specific products per holiday
fig_holiday = px.bar(
    holiday_product_demand_melted,
    x='holiday_name',
    y='Average Demand',
    color='Product',
    title='Average Demand for Cerveza and Salchipapa per Holiday',
    labels={'holiday_name': 'Holiday Name', 'Average Demand': 'Average Demand Quantity', 'Product': 'Product Type'},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_holiday.update_layout(xaxis_title='Holiday Name', yaxis_title='Average Demand')
st.plotly_chart(fig_holiday, width='stretch')

# Plot 4: Box plot of daily product costs
fig_boxplot = px.box(df_costs_melted, x='Product Cost', y='Daily Cost',
                     color='Product Cost',
                     title='Distribution of Daily Costs per Product',
                     labels={'Product Cost': 'Product', 'Daily Cost': 'Daily Cost Value'},
                     color_discrete_sequence=px.colors.qualitative.Plotly)
fig_boxplot.update_layout(xaxis_title='Product', yaxis_title='Daily Cost')
st.plotly_chart(fig_boxplot, width='stretch')


df_2025 = df[df['date'].dt.year == 2025].copy()
df_2025['month'] = df_2025['date'].dt.month

monthly_aggregated_data = df_2025.groupby('month')[product_columns + ['ganancia']].sum().reset_index()

print("Monthly aggregated demand and profit for 2025 has been calculated:")
print(monthly_aggregated_data.head())


product_profit_columns = [f'{p}_ganancia' for p in product_columns]

# Re-aggregate to include monthly profit per product
monthly_aggregated_data_detailed = df_2025.groupby('month')[product_columns + product_profit_columns].sum().reset_index()

# Melt for demand
monthly_demand_melted = monthly_aggregated_data_detailed.melt(
    id_vars='month',
    value_vars=product_columns,
    var_name='Product',
    value_name='Demand'
)

# Melt for profit
monthly_profit_melted = monthly_aggregated_data_detailed.melt(
    id_vars='month',
    value_vars=product_profit_columns,
    var_name='Product_Profit_Col',
    value_name='Profit'
)

# Clean product name in profit melted DataFrame for merging
monthly_profit_melted['Product'] = monthly_profit_melted['Product_Profit_Col'].str.replace('_ganancia', '')

# Merge demand and profit dataframes
monthly_combined_melted = pd.merge(monthly_demand_melted, monthly_profit_melted[['month', 'Product', 'Profit']], on=['month', 'Product'])

print("Monthly aggregated data for demand and profit per product has been melted and merged:")
print(monthly_combined_melted.head())




