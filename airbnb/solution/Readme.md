```python
## Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20000)
```


```python
## Import data
path = '/Users/nikhilsawal/OneDrive/machine_learning/data_science_case_studies/airbnb/solution/data/'
image_path = '/Users/nikhilsawal/OneDrive/machine_learning/data_science_case_studies/airbnb/'

contacts = pd.read_csv(path + 'contacts.csv')
listings = pd.read_csv(path + 'listings.csv')
users = pd.read_csv(path + 'users.csv')
```

# EDA


```python
print(contacts.shape)
print(listings.shape)
print(users.shape)
```

    (27887, 14)
    (13038, 4)
    (31525, 3)


# Contacts


```python
contacts.head()

ts_col = [col for col in contacts if col.startswith('ts_')]
contacts[ts_col] = contacts[ts_col].apply(pd.to_datetime)

ds_col = [col for col in contacts if col.startswith('ds_')]
contacts[ds_col] = contacts[ds_col].apply(pd.to_datetime)

summary_df = {'data_types' : contacts.dtypes,
              'missing_values' : contacts.isna().sum()}

summary_df = pd.DataFrame(summary_df)
summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_types</th>
      <th>missing_values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id_guest_anon</th>
      <td>object</td>
      <td>0</td>
    </tr>
    <tr>
      <th>id_host_anon</th>
      <td>object</td>
      <td>0</td>
    </tr>
    <tr>
      <th>id_listing_anon</th>
      <td>object</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ts_interaction_first</th>
      <td>datetime64[ns]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ts_reply_at_first</th>
      <td>datetime64[ns]</td>
      <td>2032</td>
    </tr>
    <tr>
      <th>ts_accepted_at_first</th>
      <td>datetime64[ns]</td>
      <td>11472</td>
    </tr>
    <tr>
      <th>ts_booking_at</th>
      <td>datetime64[ns]</td>
      <td>16300</td>
    </tr>
    <tr>
      <th>ds_checkin_first</th>
      <td>datetime64[ns]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ds_checkout_first</th>
      <td>datetime64[ns]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>m_guests</th>
      <td>float64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m_interactions</th>
      <td>int64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>m_first_message_length_in_characters</th>
      <td>float64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>contact_channel_first</th>
      <td>object</td>
      <td>0</td>
    </tr>
    <tr>
      <th>guest_user_stage_first</th>
      <td>object</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np

# Merge contacts and listings
merged_df = pd.merge(listings, contacts, how = 'left', on = 'id_listing_anon')

# Get listing_ids not in contacts
delta_listings = np.setdiff1d(listings['id_listing_anon'], contacts['id_listing_anon'])

# Get listing with and without checkins
no_checkins_df = merged_df[merged_df['id_listing_anon'].isin(delta_listings)]
checkins_df = merged_df[~merged_df['id_listing_anon'].isin(delta_listings)]

checkins_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_listing_anon</th>
      <th>room_type</th>
      <th>listing_neighborhood</th>
      <th>total_reviews</th>
      <th>id_guest_anon</th>
      <th>id_host_anon</th>
      <th>ts_interaction_first</th>
      <th>ts_reply_at_first</th>
      <th>ts_accepted_at_first</th>
      <th>ts_booking_at</th>
      <th>ds_checkin_first</th>
      <th>ds_checkout_first</th>
      <th>m_guests</th>
      <th>m_interactions</th>
      <th>m_first_message_length_in_characters</th>
      <th>contact_channel_first</th>
      <th>guest_user_stage_first</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71582793-e5f8-46d7-afdf-7a31d2341c79</td>
      <td>Private room</td>
      <td>-unknown-</td>
      <td>0.0</td>
      <td>8aa69e53-3bc0-4f14-bf39-1143280fd0a2</td>
      <td>01b76a9a-4c2f-4831-920b-228f2d4a55a4</td>
      <td>2016-04-03 23:12:10</td>
      <td>2016-04-04 12:11:41</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-05-20</td>
      <td>2016-05-23</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>72.0</td>
      <td>book_it</td>
      <td>new</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1a3f728-e21f-4432-96aa-361d28e2b319</td>
      <td>Entire home/apt</td>
      <td>Copacabana</td>
      <td>0.0</td>
      <td>a4023626-f62d-48ac-a7cc-4a18acf0a678</td>
      <td>74eaa99c-a053-4b7c-8cd4-d6d11c4ddf00</td>
      <td>2016-01-25 20:27:07</td>
      <td>2016-01-25 20:49:44</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-02-20</td>
      <td>2016-02-22</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>135.0</td>
      <td>contact_me</td>
      <td>past_booker</td>
    </tr>
    <tr>
      <th>2</th>
      <td>353a68be-ecf9-4b7b-9533-c882dc2f0760</td>
      <td>Entire home/apt</td>
      <td>Barra da Tijuca</td>
      <td>3.0</td>
      <td>7cabe284-b107-424a-a29f-61c78698eeb5</td>
      <td>72600771-2e0f-4502-9ae8-7ab22df6ebb8</td>
      <td>2016-06-16 01:07:42</td>
      <td>2016-06-16 01:25:46</td>
      <td>2016-06-16 01:25:48</td>
      <td>NaT</td>
      <td>2016-06-21</td>
      <td>2016-06-28</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>41.0</td>
      <td>contact_me</td>
      <td>past_booker</td>
    </tr>
    <tr>
      <th>3</th>
      <td>353a68be-ecf9-4b7b-9533-c882dc2f0760</td>
      <td>Entire home/apt</td>
      <td>Barra da Tijuca</td>
      <td>3.0</td>
      <td>b5a743fd-509e-4e44-bfb6-e03bb7185e04</td>
      <td>72600771-2e0f-4502-9ae8-7ab22df6ebb8</td>
      <td>2016-06-30 02:09:23</td>
      <td>2016-06-30 02:09:24</td>
      <td>2016-06-30 02:09:24</td>
      <td>2016-06-30 02:09:24</td>
      <td>2016-07-09</td>
      <td>2016-07-11</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>instant_book</td>
      <td>new</td>
    </tr>
    <tr>
      <th>4</th>
      <td>353a68be-ecf9-4b7b-9533-c882dc2f0760</td>
      <td>Entire home/apt</td>
      <td>Barra da Tijuca</td>
      <td>3.0</td>
      <td>1df25499-4c2c-4d96-843a-ccc338cf6874</td>
      <td>72600771-2e0f-4502-9ae8-7ab22df6ebb8</td>
      <td>2016-06-08 07:47:36</td>
      <td>2016-06-08 22:44:50</td>
      <td>2016-06-08 22:44:52</td>
      <td>NaT</td>
      <td>2016-12-15</td>
      <td>2017-01-15</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>332.0</td>
      <td>contact_me</td>
      <td>new</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Value Count
print(checkins_df['contact_channel_first'].value_counts())
print(checkins_df['guest_user_stage_first'].value_counts())
print(no_checkins_df.shape)

# Count of unique values
print(len(listings['id_listing_anon'].unique()))
print(len(contacts['id_listing_anon'].unique()))
len(checkins_df['id_listing_anon'].unique())
```

    contact_me      12828
    book_it          8366
    instant_book     6693
    Name: contact_channel_first, dtype: int64
    new            15905
    past_booker    11947
    -unknown-         35
    Name: guest_user_stage_first, dtype: int64
    (219, 17)
    13038
    12819





    12819




```python
checkins_df = checkins_df.sort_values(by = 'ds_checkin_first', ascending = True)
checkins_df['month_yr'] = pd.to_datetime(checkins_df['ts_interaction_first']).dt.to_period('M')

checkins_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_listing_anon</th>
      <th>room_type</th>
      <th>listing_neighborhood</th>
      <th>total_reviews</th>
      <th>id_guest_anon</th>
      <th>id_host_anon</th>
      <th>ts_interaction_first</th>
      <th>ts_reply_at_first</th>
      <th>ts_accepted_at_first</th>
      <th>ts_booking_at</th>
      <th>ds_checkin_first</th>
      <th>ds_checkout_first</th>
      <th>m_guests</th>
      <th>m_interactions</th>
      <th>m_first_message_length_in_characters</th>
      <th>contact_channel_first</th>
      <th>guest_user_stage_first</th>
      <th>month_yr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15783</th>
      <td>863def9e-f64d-4de3-8eca-beef4d5142c6</td>
      <td>Private room</td>
      <td>-unknown-</td>
      <td>0.0</td>
      <td>2b7e44cb-e9b4-40db-9f40-37c5e4fcb2b0</td>
      <td>f7aaf4e2-fa75-44d3-aaf7-1fd56bfc7be0</td>
      <td>2016-01-01 14:07:54</td>
      <td>2016-01-01 14:14:55</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-01-01</td>
      <td>2016-01-05</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>215.0</td>
      <td>contact_me</td>
      <td>new</td>
      <td>2016-01</td>
    </tr>
    <tr>
      <th>11045</th>
      <td>00c5ad42-75fb-4ca6-90a3-679809368a45</td>
      <td>Private room</td>
      <td>-unknown-</td>
      <td>1.0</td>
      <td>4ac626d8-2e2e-4f26-bc3c-aef3c88e67ba</td>
      <td>f541ac0e-2619-4bc6-86bd-a8299e12bb35</td>
      <td>2016-01-01 07:51:38</td>
      <td>2016-01-01 16:15:20</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-01-01</td>
      <td>2016-01-02</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>69.0</td>
      <td>book_it</td>
      <td>new</td>
      <td>2016-01</td>
    </tr>
    <tr>
      <th>10127</th>
      <td>c06110e5-256e-4e5d-bb97-a56beeea3de7</td>
      <td>Private room</td>
      <td>Copacabana</td>
      <td>26.0</td>
      <td>3350a226-eeca-429b-be2a-79d251998de7</td>
      <td>c712aaf7-6666-4f6e-8580-9a2431f314e0</td>
      <td>2016-01-01 13:57:45</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-01-01</td>
      <td>2016-01-02</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>210.0</td>
      <td>contact_me</td>
      <td>past_booker</td>
      <td>2016-01</td>
    </tr>
    <tr>
      <th>418</th>
      <td>0ed4cf7e-5fb0-4248-b2b3-fe277c22ea3a</td>
      <td>Entire home/apt</td>
      <td>Copacabana</td>
      <td>22.0</td>
      <td>05f35d89-2121-4fe2-9682-44c1ff2b6a01</td>
      <td>e7c5fd22-84f1-44a7-a7f1-f32c0462bc27</td>
      <td>2016-01-01 12:13:15</td>
      <td>2016-01-01 14:40:38</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-01-01</td>
      <td>2016-01-03</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>158.0</td>
      <td>contact_me</td>
      <td>new</td>
      <td>2016-01</td>
    </tr>
    <tr>
      <th>11046</th>
      <td>00c5ad42-75fb-4ca6-90a3-679809368a45</td>
      <td>Private room</td>
      <td>-unknown-</td>
      <td>1.0</td>
      <td>a6f9e127-4e55-42ee-9e06-0a5803414113</td>
      <td>f541ac0e-2619-4bc6-86bd-a8299e12bb35</td>
      <td>2016-01-01 11:24:25</td>
      <td>2016-01-01 16:13:54</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>2016-01-01</td>
      <td>2016-01-02</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>133.0</td>
      <td>book_it</td>
      <td>new</td>
      <td>2016-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
month_grp = checkins_df.groupby(['month_yr'])
listings = pd.DataFrame(month_grp['contact_channel_first'].value_counts())
listings.columns = ['count_listings']
listings.reset_index(inplace=True)
print(listings)
```

       month_yr contact_channel_first  count_listings
    0   2016-01            contact_me            4228
    1   2016-01               book_it            2371
    2   2016-01          instant_book            1019
    3   2016-02            contact_me            1798
    4   2016-02               book_it            1291
    5   2016-02          instant_book             982
    6   2016-03            contact_me            1640
    7   2016-03               book_it            1200
    8   2016-03          instant_book             977
    9   2016-04            contact_me            1603
    10  2016-04               book_it            1127
    11  2016-04          instant_book             986
    12  2016-05            contact_me            1761
    13  2016-05          instant_book            1334
    14  2016-05               book_it            1209
    15  2016-06            contact_me            1798
    16  2016-06          instant_book            1395
    17  2016-06               book_it            1168


# Listings by Channel type


```python
labels = 'contact_me', 'book_it', 'instant_book'
counts = checkins_df['contact_channel_first'].value_counts().values.tolist()

fig, ax = plt.subplots(figsize=(10, 7))
wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.2f%%', startangle=90)
plt.setp(autotexts, size = 15, weight ="bold")
ax.set_title('Listings by Channel Type (%)', size = 25)
plt.show()
fig.savefig(image_path + 'plots/listings.png')

```


![png](output_11_0.png)


# Time between different stages of communication


```python
from datetime import datetime

# First interaction to reply
duration = checkins_df['ts_reply_at_first'] - checkins_df['ts_interaction_first']
duration_in_s = [i for i in duration.apply(lambda x: x.total_seconds())]

interaction_to_reply = [divmod(i, 3600)[0] for i in duration_in_s]

# Reply to acceptance
duration =  checkins_df['ts_accepted_at_first'] - checkins_df['ts_reply_at_first']
duration_in_s = [i for i in duration.apply(lambda x: x.total_seconds())]

reply_to_acceptance = [divmod(i, 3600)[0] for i in duration_in_s]

# Acceptance to booking
duration =  checkins_df['ts_booking_at'] - checkins_df['ts_accepted_at_first']
duration_in_s = [i for i in duration.apply(lambda x: x.total_seconds())]

acceptance_to_booking = [divmod(i, 3600)[0] for i in duration_in_s]

# First interaction to booking
duration =  checkins_df['ts_booking_at'] - checkins_df['ts_interaction_first']
duration_in_s = [i for i in duration.apply(lambda x: x.total_seconds())]

interaction_to_booking = [divmod(i, 3600)[0] for i in duration_in_s]

```


```python
# Add the time variables to checkins_df

checkins_df['interaction_to_booking'] = interaction_to_booking
checkins_df['interaction_to_reply'] = interaction_to_reply
checkins_df['acceptance_to_booking'] = acceptance_to_booking
checkins_df['reply_to_acceptance'] = reply_to_acceptance
```

# Calculate conversion rates at different stages of funnel


```python
# Count of missing values
checkins_df['is_interaction_first'] = checkins_df['ts_interaction_first'].isna()
checkins_df['is_reply_at_first'] = checkins_df['ts_reply_at_first'].isna()
checkins_df['is_accepted_at_first'] = checkins_df['ts_accepted_at_first'].isna()
checkins_df['is_booking_at'] = checkins_df['ts_booking_at'].isna()


def get_final_df(input_df):
    
    month_grp = input_df.groupby(['month_yr'])
    
    # Make dataframes
    df_1 = pd.DataFrame(month_grp['is_interaction_first'].value_counts())
    df_2 = pd.DataFrame(month_grp['is_reply_at_first'].value_counts())
    df_3 = pd.DataFrame(month_grp['is_accepted_at_first'].value_counts())
    df_4 = pd.DataFrame(month_grp['is_booking_at'].value_counts())

    # Rename count columns
    df_1.columns = ['count_interaction_first']
    df_2.columns = ['count_reply']
    df_3.columns = ['count_accepted']
    df_4.columns = ['count_booking']

    # Reset indexes
    df_1.reset_index(inplace=True)
    df_2.reset_index(inplace=True)
    df_3.reset_index(inplace=True)
    df_4.reset_index(inplace=True)

    merged_counts_df = pd.merge(df_4, df_3, how = 'left', 
                                left_on = ['month_yr', 'is_booking_at'],
                                right_on = ['month_yr', 'is_accepted_at_first'])

    merged_counts_df_1 = pd.merge(df_2, df_1, how = 'left', 
                                  left_on = ['month_yr', 'is_reply_at_first'],
                                  right_on = ['month_yr', 'is_interaction_first'])

    merged_counts_df_1['is_interaction_first'].fillna(value=True, inplace=True)
    merged_counts_df_1['count_interaction_first'].fillna(0, inplace=True)

    final_merged_counts_df = pd.merge(merged_counts_df, merged_counts_df_1, 
                                      how = 'left',
                                      left_on = ['month_yr', 'is_booking_at'],
                                      right_on = ['month_yr', 'is_reply_at_first'])

    final_merged_counts_df = final_merged_counts_df.drop(['is_accepted_at_first', 
                                                          'is_reply_at_first',
                                                          'is_interaction_first'], 
                                                          axis = 1) 

    bookings_df = final_merged_counts_df[final_merged_counts_df['is_booking_at']==False]
    churn_df = final_merged_counts_df[final_merged_counts_df['is_booking_at']==True]

    bookings_df.columns = ['month_yr', 'is_booking_at', 'booking(#)', 
                        'interaction_accepted(#)', 'interaction_reply(#)', 
                        'interaction_started(#)']

    churn_df.columns = ['month_yr', 'is_booking_at', 'churned_at_booking(#)', 
                        'interaction_rejected(#)', 'no_reply(#)', 
                        'interaction_started(#)']

    bookings_df = bookings_df.drop(['is_booking_at'], axis=1)
    churn_df = churn_df.drop(['is_booking_at', 'interaction_started(#)'], axis=1)

    bookings_df['interaction_started(#)'] = [i for i in bookings_df['interaction_started(#)'].apply(lambda x: int(x))]
    final_df = pd.merge(bookings_df, churn_df.iloc[:,:2], how = 'left', on = 'month_yr')
    
    final_df['reply_rate(%)'] = round(final_df['interaction_reply(#)'] / final_df['interaction_started(#)'] * 100, 2)
    final_df['booking_rate(%)'] = round(final_df['booking(#)'] / final_df['interaction_started(#)'] * 100, 2)
    final_df['abandonment_rate(%)'] = round(final_df['churned_at_booking(#)'] / final_df['interaction_started(#)'] * 100, 2)
    
    return final_df
```

# Conversion over time


```python
final_df = get_final_df(checkins_df)
```


```python
x = [i for i in final_df['month_yr'].apply(lambda x: x.strftime('%Y-%b'))]
x_indexes = np.arange(len(x))
width = 0.15
fig, ax = plt.subplots()
fig.set_figheight(20)
fig.set_figwidth(30)

# plt.style.use("fivethirtyeight")
rects1 = ax.bar(x_indexes, 
         final_df['interaction_started(#)'], 
         width=width,
         color="#FF5A5F", 
         label="Total first interaction over time")
rects2 = ax.bar(x_indexes + 0.15, 
         final_df['interaction_reply(#)'],
         width=width,
         color="#00A699", 
         label="Total replys over time")
rects3 = ax.bar(x_indexes + 0.3, 
         final_df['interaction_accepted(#)'],
         width=width,
         color="#484848", 
         label="Total accepted requests over time")
rects4 = ax.bar(x_indexes + 0.45, 
         final_df['booking(#)'],
         width=width,
         color="#767676", 
         label="Total Bookings over time")
rects5 = ax.bar(x_indexes + 0.6, 
         final_df['churned_at_booking(#)'],
         width=width,
         color="#FC642D", 
         label="Total Bookings churned over time")
plt.legend(("Total first interaction over time", "Total replys over time", 
            "Total accepted requests over time", "Total Bookings over time",
            "Total Bookings churned over time"), fontsize=20)
plt.xticks(ticks=x_indexes + 2.00*width, labels=x, fontsize=30)
plt.yticks(fontsize=30)
plt.suptitle("Overall conversion over time", fontsize=50, ha='center')
plt.title("First Interaction / First Reply / Request Accepted / Bookings / Abandonment", fontsize=20)
plt.xlabel("Month-yr", fontsize=40)
plt.ylabel("# Count", fontsize=40)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(9, 10),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

plt.show()
```


![png](output_19_0.png)


# Conversion rates over time


```python
from matplotlib.ticker import FormatStrFormatter

x = [i for i in final_df['month_yr'].apply(lambda x: x.strftime('%Y-%m'))]
x_indexes = np.arange(len(x))
y_indexes = np.arange(0, 110, 10)
width = 0.25

fig, ax = plt.subplots()
fig.set_figheight(20)
fig.set_figwidth(30)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f%%'))
rects1 = ax.bar(x_indexes + 0.25, 
         final_df['abandonment_rate(%)'],
         width=width,
         color="#484848", 
         label="Abandonment Rate (%)")
rects2 = ax.bar(x_indexes + 0.5, 
         final_df['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")
plt.legend(("Abandonment Rate (%)", "Conversion Rate (%)"), fontsize=25)
plt.xticks(ticks=x_indexes + 1.5*width, labels=x, fontsize=20)
plt.yticks(ticks=y_indexes, fontsize=20)
plt.title("As percentage of number of Interaction Started", fontsize=30, ha='center')
plt.suptitle("Overall conversion/abandonment rate", fontsize=50, ha='center')
plt.xlabel("Month-yr", fontsize=40)
plt.ylabel("Conversion Rate (%)", fontsize=40)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} %'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(18, 15),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20)

autolabel(rects1)
autolabel(rects2)

plt.show()
```


![png](output_21_0.png)


# Conversion rate by Room type


```python
checkins_df['room_type'].unique()

checkins_df_private_room = checkins_df[checkins_df['room_type'] == 'Private room']
checkins_df_private_room = get_final_df(checkins_df_private_room)

checkins_df_entire_apt = checkins_df[checkins_df['room_type'] == 'Entire home/apt']
checkins_df_entire_apt = get_final_df(checkins_df_entire_apt)

checkins_df_shared_room = checkins_df[checkins_df['room_type'] == 'Shared room']
checkins_df_shared_room = get_final_df(checkins_df_shared_room)
```


```python
from matplotlib.ticker import PercentFormatter

x = [i for i in checkins_df_private_room['month_yr'].apply(lambda x: x.strftime('%Y-%m'))]
x_indexes = np.arange(len(x))
y_indexes = np.arange(0, 110, 10)
width = 0.20

fig, ax = plt.subplots(3,1)
fig.set_figheight(30)
fig.set_figwidth(30)
top_ax, middle_ax, bottom_ax = ax

################
# Private Room #
################

top_ax.bar(x_indexes + 0.00, 
         checkins_df_private_room['abandonment_rate(%)'],
         width=width,
         color="#FF5A5F", 
         label="Abandonment Rate (%)")
top_ax.bar(x_indexes + 0.20, 
         checkins_df_private_room['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

# plt.xticks(ticks=x_indexes + 0.5*width, labels=x, fontsize=20)
top_ax.legend(fontsize=30, loc='upper right')
top_ax.set_title('(Private Room)', fontsize=30)
top_ax.set_xticks(ticks=x_indexes + 0.5*width)
top_ax.set_xticklabels(labels=x, fontsize=30)

top_ax.set_yticks(ticks=y_indexes)
top_ax.set_yticklabels(y_indexes, fontsize = 30)

####################
# Entire Apartment #
####################

middle_ax.bar(x_indexes + 0.00, 
         checkins_df_entire_apt['abandonment_rate(%)'],
         width=width,
         color="#FF5A5F", 
         label="Abandonment Rate (%)")
middle_ax.bar(x_indexes + 0.20, 
         checkins_df_entire_apt['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

middle_ax.legend(fontsize=30, loc='upper right')
middle_ax.set_xticks(ticks=x_indexes + 0.5*width)
middle_ax.set_xticklabels(labels=x, fontsize=30)
middle_ax.set_yticks(ticks=y_indexes)
middle_ax.set_yticklabels(y_indexes, fontsize = 30)
middle_ax.set_title('(Entire Apartment)', fontsize=30)

###############
# Shared Room #
###############

bottom_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f%%'))
bottom_ax.bar(x_indexes + 0.00, 
         checkins_df_shared_room['abandonment_rate(%)'],
         width=width,
         color="#FF5A5F", 
         label="Abandonment Rate (%)")
bottom_ax.bar(x_indexes + 0.20, 
         checkins_df_shared_room['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

bottom_ax.legend(fontsize=30, loc='upper right')
bottom_ax.set_xticks(ticks=x_indexes + 0.5*width)
bottom_ax.set_xticklabels(labels=x, fontsize=30)
bottom_ax.set_yticks(ticks=y_indexes)
bottom_ax.set_yticklabels(y_indexes, fontsize=30)
bottom_ax.set_title('(Shared Room)', fontsize=30)

fig.text(0.5, 0.01, 'Month-Yr', ha='center', fontsize=40)
fig.text(0.03, 0.5, 'As Percentage of Interaction Started (%)', 
         va='center', rotation='vertical', fontsize=40)

plt.suptitle("Overall conversion/abandonment rate - Apartment Type", fontsize=50, ha='center')

plt.show()
fig.savefig(image_path + 'plots/conversion_by_room_type.png')

# 1. Shared rooms are more likely to be abandoned over time. Typically shared
# rooms are not favourites with families. So based on this analysis we might 
# want to look into our targeting strategies for shared rooms.

# 2. Also, the abandonment in the month of January is consistantly high. So
# we might want to look into that.
```


![png](output_24_0.png)


# Distribution of length of first message


```python
from statistics import mean

bookings_df = checkins_df[checkins_df['is_booking_at']==False]
abandonment_df = checkins_df[checkins_df['is_booking_at']==True]

bookings_df = bookings_df[bookings_df['m_first_message_length_in_characters']<800]
abandonment_df = abandonment_df[abandonment_df['m_first_message_length_in_characters']<800]

plt.hist(abandonment_df['m_first_message_length_in_characters'], edgecolor='black', bins=50, alpha=0.7, label='Abandonments')
plt.hist(bookings_df['m_first_message_length_in_characters'], edgecolor='black', bins=50, alpha=0.7, label='Bookings')
plt.axvline(abandonment_df['m_first_message_length_in_characters'].mean(), color='blue', linestyle='dashed', linewidth=3)
plt.axvline(bookings_df['m_first_message_length_in_characters'].mean(), color='orange', linestyle='dashed', linewidth=3)
plt.xlim(0, 800)
plt.xlabel("Message length (in characters)", size=30)
plt.ylabel("Count", size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.title("Length of first message (Bookings Vs. Abandoned)", size=40)
plt.rcParams["figure.figsize"] = (20,15)
plt.legend(loc='upper right', fontsize=20)
plt.savefig(image_path + 'plots/first_inter_length_dist.png')

```


![png](output_26_0.png)


# Distribution of number of interactions


```python
bookings_df = checkins_df[checkins_df['is_booking_at']==False]
abandonment_df = checkins_df[checkins_df['is_booking_at']==True]

bookings_df = bookings_df[bookings_df['m_interactions']<30]
abandonment_df = abandonment_df[abandonment_df['m_interactions']<30]

bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

plt.hist(abandonment_df['m_interactions'], edgecolor='black', bins=bins, alpha=0.7, label='Abandonments')
plt.hist(bookings_df['m_interactions'], edgecolor='black', bins=bins, alpha=0.7, label='Bookings')
plt.axvline(abandonment_df['m_interactions'].mean(), color='blue', linestyle='dashed', linewidth=3)
plt.axvline(bookings_df['m_interactions'].mean(), color='orange', linestyle='dashed', linewidth=3)
plt.xlim(0, 27)
plt.xlabel("Number of interactions", size=30)
plt.ylabel("Count", size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.title("Interactions distribution (Bookings Vs. Abandoned)", size=40)
plt.rcParams["figure.figsize"] = (20,15)
plt.legend(loc='upper right', fontsize=20)
plt.savefig(image_path + 'plots/count_interaction_dist.png')

```


![png](output_28_0.png)


# Conversion rate by Contact Channels


```python
checkins_df['contact_channel_first'].unique()

checkins_df_contact_me = checkins_df[checkins_df['contact_channel_first'] == 'contact_me']
checkins_df_contact_me = get_final_df(checkins_df_contact_me)

checkins_df_book_it = checkins_df[checkins_df['contact_channel_first'] == 'book_it']
checkins_df_book_it = get_final_df(checkins_df_book_it)

checkins_df_instant_book = checkins_df[checkins_df['contact_channel_first'] == 'instant_book']
checkins_df_instant_book = get_final_df(checkins_df_instant_book)
```


```python
x = [i for i in checkins_df_book_it['month_yr'].apply(lambda x: x.strftime('%Y-%m'))]
x_indexes = np.arange(len(x))
y_indexes = np.arange(0, 110, 10)
width = 0.20

fig, ax = plt.subplots(2,1)
fig.set_figheight(30)
fig.set_figwidth(30)
top_ax, bottom_ax = ax

###########
# Book It #
###########

top_ax.bar(x_indexes + 0.00, 
         checkins_df_book_it['abandonment_rate(%)'],
         width=width,
         color="#FF5A5F", 
         label="Abandonment Rate (%)")
top_ax.bar(x_indexes + 0.20, 
         checkins_df_book_it['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

# plt.xticks(ticks=x_indexes + 0.5*width, labels=x, fontsize=20)
top_ax.legend(fontsize=30, loc='upper right')
top_ax.set_title('(Book It)', fontsize=30)
top_ax.set_xticks(ticks=x_indexes + 0.5*width)
top_ax.set_xticklabels(labels=x, fontsize=30)

top_ax.set_yticks(ticks=y_indexes)
top_ax.set_yticklabels(y_indexes, fontsize = 30)

##############
# Contact Me #
##############

bottom_ax.bar(x_indexes + 0.00, 
         checkins_df_contact_me['abandonment_rate(%)'],
         width=width,
         color="#FF5A5F", 
         label="Abandonment Rate (%)")
bottom_ax.bar(x_indexes + 0.20, 
         checkins_df_contact_me['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

bottom_ax.set_xticks(ticks=x_indexes + 0.5*width)
bottom_ax.set_xticklabels(labels=x, fontsize=30)
bottom_ax.set_yticks(ticks=y_indexes)
bottom_ax.set_yticklabels(y_indexes, fontsize = 30)
bottom_ax.set_title('(Contact me)', fontsize=30)
bottom_ax.legend(fontsize=30, loc='upper right')

fig.text(0.5, 0.01, 'Month-Yr', ha='center', fontsize=40)
fig.text(0.03, 0.5, 'As percentage of number of interaction started (%)', 
         va='center', rotation='vertical', fontsize=40)

plt.suptitle("Conversion/abandonment rate - Contact channel", fontsize=50, ha='center')

plt.show()
fig.savefig(image_path + 'plots/contact_channel_aban_conv_rate.png')

```


![png](output_31_0.png)


# Dataframe for Time between interactions by User type


```python
# Contact Me
contact_me_df = checkins_df[checkins_df['contact_channel_first']=='contact_me']
buyers_grp = contact_me_df.groupby(['guest_user_stage_first'])

start_to_finish = pd.DataFrame(buyers_grp['interaction_to_booking'].mean())
start_to_finish.columns = ['start-to-finish']
start_to_finish.reset_index(inplace=True)

interaction_to_reply = pd.DataFrame(buyers_grp['interaction_to_reply'].mean())
interaction_to_reply.columns = ['interaction-to-reply']
interaction_to_reply.reset_index(inplace=True)

reply_to_acceptance = pd.DataFrame(buyers_grp['reply_to_acceptance'].mean())
reply_to_acceptance.columns = ['reply_to_acceptance']
reply_to_acceptance.reset_index(inplace=True)

acceptance_to_booking = pd.DataFrame(buyers_grp['acceptance_to_booking'].mean())
acceptance_to_booking.columns = ['acceptance_to_booking']
acceptance_to_booking.reset_index(inplace=True)

contact_me_final_df = pd.concat([start_to_finish, interaction_to_reply, 
                                 reply_to_acceptance, acceptance_to_booking], axis='columns').iloc[1:,[0,1,3,5,7]]

# Book It
book_it_df = checkins_df[checkins_df['contact_channel_first']=='book_it']
buyers_grp = book_it_df.groupby(['guest_user_stage_first'])

start_to_finish = pd.DataFrame(buyers_grp['interaction_to_booking'].mean())
start_to_finish.columns = ['start-to-finish']
start_to_finish.reset_index(inplace=True)

interaction_to_reply = pd.DataFrame(buyers_grp['interaction_to_reply'].mean())
interaction_to_reply.columns = ['interaction-to-reply']
interaction_to_reply.reset_index(inplace=True)

reply_to_acceptance = pd.DataFrame(buyers_grp['reply_to_acceptance'].mean())
reply_to_acceptance.columns = ['reply_to_acceptance']
reply_to_acceptance.reset_index(inplace=True)

acceptance_to_booking = pd.DataFrame(buyers_grp['acceptance_to_booking'].mean())
acceptance_to_booking.columns = ['acceptance_to_booking']
acceptance_to_booking.reset_index(inplace=True)

book_it_final_df = pd.concat([start_to_finish, interaction_to_reply, 
                                 reply_to_acceptance, acceptance_to_booking], axis='columns').iloc[1:,[0,1,3,5,7]]

final_df = pd.concat([contact_me_final_df, book_it_final_df], axis='rows')
final_df['contact_channel'] = ['contact_me', 'contact_me', 'book_it', 'book_it']


new_booker = final_df[final_df['guest_user_stage_first']=='new']
new_booker = new_booker.iloc[:,1:]
past_booker = final_df[final_df['guest_user_stage_first']=='past_booker']
past_booker = past_booker.iloc[:,1:]
```

# Time between interactions by User type


```python
x = past_booker.columns[:4]
x_indexes = np.arange(len(x))
y_indexes = np.arange(0, 110, 10)
width = 0.20

fig, ax = plt.subplots(2,1)
fig.set_figheight(30)
fig.set_figwidth(30)
top_ax, bottom_ax = ax

###############
# New Bookers #
###############

top_ax.bar(x_indexes + 0.00, 
         new_booker[new_booker['contact_channel']=='contact_me'].values.tolist()[0][:4],
         width=width,
         color="#FF5A5F", 
         label="Contact me")
top_ax.bar(x_indexes + 0.20, 
         new_booker[new_booker['contact_channel']=='book_it'].values.tolist()[0][:4],
         width=width,
         color="#767676", 
         label="Book it")

# plt.xticks(ticks=x_indexes + 0.5*width, labels=x, fontsize=20)
top_ax.legend(fontsize=30, loc='upper right')
top_ax.set_title('New Bookers', fontsize=35)
top_ax.set_xticks(ticks=x_indexes + 0.5*width)
top_ax.set_xticklabels(labels=x, fontsize=30)

top_ax.set_yticks(ticks=y_indexes)
top_ax.set_yticklabels(y_indexes, fontsize = 30)

################
# Past Bookers #
################

bottom_ax.bar(x_indexes + 0.00, 
         past_booker[past_booker['contact_channel']=='contact_me'].values.tolist()[0][:4],
         width=width,
         color="#FF5A5F", 
         label="Contact me")
bottom_ax.bar(x_indexes + 0.20, 
         past_booker[past_booker['contact_channel']=='book_it'].values.tolist()[0][:4],
         width=width,
         color="#767676", 
         label="Book it")

bottom_ax.set_xticks(ticks=x_indexes + 0.5*width)
bottom_ax.set_xticklabels(labels=x, fontsize=30)
bottom_ax.set_yticks(ticks=y_indexes)
bottom_ax.set_yticklabels(y_indexes, fontsize = 30)
bottom_ax.set_title('Past Bookers', fontsize=35)
bottom_ax.legend(fontsize=30, loc='upper right')


fig.text(0.5, 0.01, 'Interaction Phases', ha='center', fontsize=40)
fig.text(0.03, 0.5, 'Average time between interactions (in Hrs)', 
         va='center', rotation='vertical', fontsize=40)

plt.suptitle("Time spent in between 'Contact Me' interactions", fontsize=50, ha='center')

plt.show()
fig.savefig(image_path + 'plots/time_spent_contact_me.png')

```


![png](output_35_0.png)


# New vs Returning users over time


```python
month_grp = checkins_df.groupby(['month_yr'])
new_returning = pd.DataFrame(month_grp['guest_user_stage_first'].value_counts())
new_returning.columns = ['count_listings']
new_returning.reset_index(inplace=True)
new_returning = new_returning.pivot_table(index='month_yr', 
                          columns='guest_user_stage_first',
                          values='count_listings')
new_returning.reset_index(inplace=True)
new_returning.reset_index(inplace=True)
new_returning.drop(columns='index', axis=1, inplace=True)
new_returning

x = [i for i in new_returning['month_yr'].apply(lambda x: x.strftime('%Y-%m'))]

plt.plot(x, new_returning['new'], label='New Bookers')
plt.plot(x, new_returning['past_booker'], label='Returning Bookers')
plt.xlabel('Month-Yr', fontsize=40)
plt.ylabel('# Bookers', fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=40)
plt.rcParams["figure.figsize"] = (20,15)
plt.show()
```


![png](output_37_0.png)


# Conversion rates by user types


```python
checkins_df['guest_user_stage_first'].unique()

checkins_df_new_booker = checkins_df[checkins_df['guest_user_stage_first'] == 'new']
checkins_df_new_booker = get_final_df(checkins_df_new_booker)

checkins_df_past_booker = checkins_df[checkins_df['guest_user_stage_first'] == 'past_booker']
checkins_df_past_booker = get_final_df(checkins_df_past_booker)
```


```python
x = [i for i in checkins_df_new_booker['month_yr'].apply(lambda x: x.strftime('%Y-%m'))]
x_indexes = np.arange(len(x))
y_indexes = np.arange(0, 110, 10)
width = 0.20

fig, ax = plt.subplots(2,1)
fig.set_figheight(30)
fig.set_figwidth(30)
top_ax, bottom_ax = ax

###############
# New Bookers #
###############

top_ax.bar(x_indexes + 0.00, 
         checkins_df_new_booker['abandonment_rate(%)'],
         width=width,
         color="#484848", 
         label="Abandonment Rate (%)")
top_ax.bar(x_indexes + 0.20, 
         checkins_df_new_booker['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

# plt.xticks(ticks=x_indexes + 0.5*width, labels=x, fontsize=20)
top_ax.legend(fontsize=30, loc='upper right')
top_ax.set_title('(New Bookers)', fontsize=30)
top_ax.set_xticks(ticks=x_indexes + 0.5*width)
top_ax.set_xticklabels(labels=x, fontsize=30)

top_ax.set_yticks(ticks=y_indexes)
top_ax.set_yticklabels(y_indexes, fontsize = 30)

#####################
# Returning Bookers #
#####################

bottom_ax.bar(x_indexes + 0.00, 
         checkins_df_past_booker['abandonment_rate(%)'],
         width=width,
         color="#484848", 
         label="Abandonment Rate (%)")
bottom_ax.bar(x_indexes + 0.20, 
         checkins_df_past_booker['booking_rate(%)'],
         width=width,
         color="#00A699", 
         label="Booking Rate (%)")

bottom_ax.set_xticks(ticks=x_indexes + 0.5*width)
bottom_ax.set_xticklabels(labels=x, fontsize=30)
bottom_ax.set_yticks(ticks=y_indexes)
bottom_ax.set_yticklabels(y_indexes, fontsize = 30)
bottom_ax.set_title('(Returning Bookers)', fontsize=30)

fig.text(0.5, 0.01, 'Month-Yr', ha='center', fontsize=40)
fig.text(0.03, 0.5, 'As Percentage of Interaction Started (%)', 
         va='center', rotation='vertical', fontsize=40)

plt.suptitle("Overall conversion/abandonment rate - Guest Type", fontsize=50, ha='center')

plt.show()


# Since booking rate for Returning buyers is better than abandonment rate
# we are doing a good job with the targeting. 
```


![png](output_40_0.png)



```python
# 1. What key metrics would you propose to monitor over time the success 
# of the team's efforts in improving the guest host matching process and 
# why? Clearly define your metric(s) and explain how each is computed.

## Key Metrics
#  1.0 % Conversion
#    1.1 

```


```python
# 2. What areas should we invest in to increase the number of successful
# bookings in Rio de Janeiro? What segments are doing well and what could 
# be improved? Propose 2-3 specific recommendations (business initiatives 
# and product changes) that could address these opportunities. Demonstrate
# rationale behind each recommendation AND prioritize your recommendations 
# in order of their estimated impact.


```


```python
# 3. There is also interest from executives at Airbnb about the work you 
# are doing, and a desire to understand the broader framing of the challenge 
# of matching supply and demand, thinking beyond the data provided. What 
# other research, experiments, or approaches could help the company get 
# more clarity on the problem?


```
