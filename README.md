# Case Studies
## 1. Airbnb - [Code Link](https://github.com/NikhilSawal/data_science_case_studies/tree/master/airbnb/plots)

### Problem
The product and operations team at Airbnb wants a detailed analysis of their guest host matching system in an attempt to grow bookings at Rio. The team wants to
1. Track a few KPI's that would help them monitor the success of the teams efforts in improving guest host matching.

2. Gain a deeper understanding of what areas to invest. Come up with a few recommendations, business initiative and product changes and demonstrate the rationale behind the recommendations.

### Analysis

#### 1. KPI's

1. Conversion rate (%)

   * Across different stages of communication between guest & host, from **`first interation -> first reply -> acceptance -> booking/abandoned`**
   * Across contact channel types (contact_me, book_it, instant_book)
   * Across user type (New vs. Returning)
   * Across room types (Private room vs. Entire Apt vs. Shared room)


2. Average time between interactions

    * Across user-type and channel-type


3. \# Interactions and # Characters in guests opening remark (Bookings vs. Abandoned)

#### 2. Recommendations

1. **Make suggestions to keep the interactions engaging**

    On an overall level, there is a massive drop in the number of accepted requests once we get past the reply phase. Providing sample suggestions to both guest and host could improve the chances of booking.

![img_1](plots/airbnb/conversion_over_time.png)

The following distributions plots provide further evidence to come up with measures to improve engagement.
1. To make a good first impression, the opening message should have no more than ~180 characters.

2. Also all the successful bookings have shown little over 10 interactions.

![img_1](plots/airbnb/first_inter_length_dist.png)

![img_1](plots/airbnb/count_interaction_dist.png)

2. **"Contact_me" performs poorly**

    Looking at the abandonment rate in the figure below, we see that "Contact_me" channel type has 10 times the abandonment rate as compared to "book_it".

    >*Note: The abandonment rate is a percentage of the total number of interactions started.*


![img_2](plots/airbnb/contact_channel_aban_conv_rate.png)


Digging a bit deeper, the following plot shows the time spent in number of hours between different phases of the interaction for new vs. returning buyers. We clearly see that average time for "Contact_me" between the phases is well over a day for both types of buyers, which provides further evidence against the channel.


![img_3](plots/airbnb/time_spent_contact_me.png)
