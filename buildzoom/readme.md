# Buildzoom - [Machine Learning Model](https://github.com/NikhilSawal/data_science_case_studies/blob/master/buildzoom/solution.ipynb)

## Problem Statement

Buildzoom gets data on building permits and wants to build a classifier that can correctly identify of the permit. The permit maybe of several types, but Buildzoom, wants a binary classification that can identify if a permit is ```(ELECTRICAL/NON-ELECTRICAL).```

## Input Data
| Data Column | Description |
| ----------- | ----------- |
| License Type | Types of license associated with the property (Electrical contractor license, Speciality contractor license) |
| Business Name | Name of business |
| Legal Description | Legal address/block information |
| Description | describes work that need to be done eg: Install low voltage security system |
| Type (y) | ELECTRICAL/NON-ELECTRICAL |
| Subtype | Commercial/Single Family |
| Job Value | Value associated with the job |

```python
import pandas as pd
import numpy as np

from statistics import median, mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

pd.set_option('display.max_rows', 90000)
pd.set_option('display.max_columns', 50)

path = '/Users/nikhilsawal/OneDrive/machine_learning/data_science_case_studies/buildzoom/data/'
```


```python
X_train = pd.read_table(path + 'train_data.csv')
X_test = pd.read_table(path + 'xtest_data.csv')
y_test = pd.read_csv(path + 'ytest_pred.csv')
```


```python
X_train.head()
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
      <th>licensetype</th>
      <th>businessname</th>
      <th>legaldescription</th>
      <th>description</th>
      <th>type</th>
      <th>subtype</th>
      <th>job_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>ADT LLC</td>
      <td>NaN</td>
      <td>INSTALL LOW VOLTAGE SECURITY SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>INSTALL (1) NON-ILLUMINATED WALL SIGN - PLAZA ...</td>
      <td>SIGN/BILLBOARD</td>
      <td>COMMERCIAL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SPECIALTY CONTRACTOR LICENSE</td>
      <td>KLN MEDIA LLC</td>
      <td>NaN</td>
      <td>INSTALL (1) NON-ILLUM ON-PREMISES WALL SIGN FO...</td>
      <td>SIGN/BILLBOARD</td>
      <td>COMMERCIAL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONT...</td>
      <td>OLSON ENERGY SERVICE</td>
      <td>NaN</td>
      <td>REPLACE OIL FURNACE W/ NEW GAS FURNACE</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WIRE NEW SINGLE FAMILY RESIDENCE W/ 200 AMP SE...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.dtypes
```




    licensetype         object
    businessname        object
    legaldescription    object
    description         object
    type                object
    subtype             object
    job_value           object
    dtype: object



# License Type


```python
print("Count of Uniques: ", len(X_train['licensetype'].unique()))
print("20 Unique values: ", X_train['licensetype'].unique()[:20])

```

    Count of Uniques:  121
    20 Unique values:  [nan 'SPECIALTY CONTRACTOR LICENSE'
     'ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC., GENERAL'
     'GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER CONTRACTOR, GENERAL CONTRACTOR LICENSE'
     'ELEVATOR CONTRACTOR LICENSE' 'ELECTRICAL CONTRACTOR LICENSE'
     'GENERAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC.'
     'ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC.'
     'GENERAL CONTRACTOR LICENSE'
     'ELECTRICAL CONTRACTOR LICENSE, GENERAL CONTRACTOR LICENSE'
     'REFRIGERATION CONTRACTOR LIC., ELECTRICAL CONTRACTOR LICENSE, GENERAL CONTRACTOR LICENSE'
     'GENERAL CONTRACTOR LICENSE, GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER CONTRACTOR'
     'GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER CONTRACTOR'
     'ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC., REGIS. SIDE SEWER CONTRACTOR'
     'REFRIGERATION CONTRACTOR LIC., GENERAL CONTRACTOR LICENSE, ELECTRICAL CONTRACTOR LICENSE'
     'ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC., GENERAL CONTRACTOR LICENSE, REFRIGERAT'
     'ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONTRACTOR LICENSE'
     'REGIS. SIDE SEWER CONTRACTOR'
     'ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC., SPECIALTY CONTRACTOR LICENSE'
     'REGIS. SIDE SEWER CONTRACTOR, GENERAL CONTRACTOR LICENSE']


**NOTE:** From the output above we see that there are **121** unique values for `licensetype`. Also, each row is a combination of different licensetypes. For ex: the third row contains 4 licenses namely `ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION CONTRACTOR LIC., GENERAL` where `ELECTRICAL CONTRACTOR LICENSE` is repeated twice, which just seems repeatative and needs to be cleaned up!!!

**ALSO NOTE:** The license type `GENERAL CONTRACTOR LICENSE` appears in different formats like `GENERAL`, `GENERAL CONTRACTOR LICENSE` and so on. So, we are going to write a function `get_pattern()` that identifies these patterns and replaces them with the correct string, using another function `clean_license()`. In the example above, if we see `GENERAL`, we would want to replace it with `GENERAL CONTRACTOR LICENSE`.


```python
import re

def get_uniques(df, col_name):

    uniques = []
    for i in df[col_name]:
        uniques += i
    return list(set(uniques))


def get_pattern(df):

    uniques = get_uniques(df, 'licensetype')
    unique = []
    for i in range(len(uniques)):

        pattern = re.compile('^'+uniques[i][:5]+'*')
        matches = []
        for index, license in enumerate(uniques):

            if pattern.search(license) is not None:
                matches.append(license)

        if matches not in unique:
            unique.append(matches)
        pass

    licenseList = []
    licenseDict = {}

    for licenses in unique:
        licenseList.append(licenses)
        licenseDict[max(licenses)] = licenses

    return licenseList
```


```python
# license type
X_train.loc[:,'licensetype'] = X_train.loc[:,'licensetype'].fillna('None')
X_train.loc[:,'licensetype'] = X_train.loc[:,'licensetype'].apply(lambda x: x.split(', ')).apply(lambda x: list(set(x)))   
pattern = get_pattern(X_train)
print(pattern[0])

```

    ['GENERAL CO', 'GENERAL C', 'GENERA', 'GENERAL CONTRACTOR LICENSE', 'GENERAL ', 'GENERAL CONT', 'GENERAL CONTRA', 'GENERAL']


The above output shows the different ways in which `GENERAL CONTRACTOR LICENSE` appears in `licensetype`. We will write a function `clean_license()` that replaces any other appearance like `'GENERAL CO', 'GENERAL C', 'GENERA'` with the complete license name `GENERAL CONTRACTOR LICENSE`.


```python
# Clean licensetype
def clean_license(inp_list, reference):

    temp_list = []
    for i in inp_list:
        if i == 'None':
            temp_list.append(i)
        else:
            for j in reference:
                temp = []
                if i in j:
                    temp_list.append(max(j).lower().replace(" ", "_"))
                    break
    return temp_list


```


```python
cleaned_license = [clean_license(item, pattern) for item in X_train['licensetype']]
print("Original Format: ", X_train['licensetype'][3],"\n")
print("Cleaned Format: ", cleaned_license[3])

```

    Original Format:  ['ELECTRICAL CONTRACTOR LICENSE', 'GENERAL', 'REFRIGERATION CONTRACTOR LIC.']

    Cleaned Format:  ['electrical_contractor_license', 'general_contractor_license', 'refrigeration_contractor_lic.']


From the output above we can now see that `GENERAL` is replaced with `general_contractor_license`. Futher we will sort each of these lists in alphabetical order and concatenate each elements of the list which should reduce the number of uniques and make `OneHotEncoding` easy for our ML model to ingest


```python
cleaned_license = ['-'.join(sorted(i)) for i in cleaned_license]
print("Formated String: ", cleaned_license[3])
X_train.loc[:,'licensetype'] = cleaned_license

```

    Formated String:  electrical_contractor_license-general_contractor_license-refrigeration_contractor_lic.



```python
print("Unique LicenseTypes: ", len(X_train['licensetype'].unique()))
```

    Unique LicenseTypes:  48


**NOTE:** We have considerabely reduced the number of unique licensetypes from `121` to `48`.

# Business Name


```python
print("Count of Uniques: ", len(X_train['businessname'].unique()))
print("", X_train['businessname'].unique())

```

    Count of Uniques:  17398
     ['ADT LLC' nan 'KLN MEDIA LLC' ... 'KRISTINE LOGAN'
     'WESCO CONSTRUCTION DIVISION' 'FRED STAUSS']



```python
def get_businessname(data, n):
    """Set top N businessnames as factor"""
    data['businessname'].fillna('None', inplace=True)
    temp = data['businessname'].value_counts().head(n).index.values
    top_n = [i.lower().replace(" ","_") if i in temp else 'Other' for i in data['businessname']]
    return top_n
```


```python
X_train.loc[:,'businessname'] = get_businessname(X_train, 100)
print("Uniques: ", X_train['businessname'].unique())
print("Uniques: ", len(X_train['businessname'].unique()))

```

    Uniques:  ['adt_llc' 'none' 'Other' 'olson_energy_service'
     'metropolitan_sewer_service_llc' 'thyssenkrupp_elevator_corp'
     'merit_mechanical_inc' 'boyer_electric_company_inc' 'kone_inc'
     'blue_flame_llc' 'guardian_security_systems_inc'
     'prime_electric_incorporated' 'active_engineering'
     'reed_wright_htg_&_elec_co_inc' 'roto_rooter_services_company'
     'titan_electric_inc' 'valley_elec_co_of_mtvernon_inc'
     'evergreen_refrigeration_llc' "o'neill_plumbing_co"
     "mr_rooter_plumbing_(sposari's)" 'kemly_electric_incorporated'
     'otis_elevator_company' 'rescue_rooter_llc' 'schindler_elevator_company'
     'north_star_electric_inc' 'sewer_friendly_llc'
     'jim_dandy_sewer_services_inc' 'cardinal_heating_&_a/c_inc'
     'adt_security_services_inc' 'allied_fire_&_security' 's_e_s_incorporated'
     'c_&_r_electric_incorporated' 'pride_electric_incorporated' 'e_h_s_llc'
     's_m_e_incorporated_of_seattle' 'm_m_comfort_systems'
     'aces_four_construction_company' 'select_air_service'
     'best_plumbing_group_llc' 'bowie_electric_svc&supply_inc'
     'brennan_heating_&_a/c_llc' 'south_west_plbg_&_wtrhtrs_inc'
     'beacon_plumbing_&_mechanical' 'washington_energy_svcs_(wesco)'
     'cochran_incorporated' 'fischer_plumbing_co_inc,_the'
     'ballard_natural_gas_svcs_inc' 'macdonald-miller_service_inc'
     'cfm_heating_&_cooling' 'sequoyah_electric_llc'
     'brennan_heating_company_inc' 'hermanson_company_llp'
     'mckinstry_company_llc' "bob's_heating_&_a/c_inc"
     'rossoe_energy_systems_inc' 'tube_art_displays_inc'
     'nelson_electric_incorporated' 'veca_electric_company_inc'
     'mckinstry_electric' 'major_electric_incorporated' 'as_you_wish_electric'
     'greenwood_heating_and_ac' 'puget_sound_solar_llc'
     'pipeline_industries_inc' 'snohomish_electric_inc'
     'brinks_home_security_inc' 'evergreen_power_systems' 'holaday_parks_inc'
     'sasco_electric' 'r_f_i_comm_&_security_systems'
     'convergint_technologies_llc' 'seatac_electric_incorporated'
     'protection_one_alarm_inc' 'emerald_aire_incorporated'
     'holmes_electric_company' 'mauro_electric_inc' 'radford_electric_company'
     'simplexgrinnell_lp' 'l_&_s_electrical_llc' 'provident_electric_inc'
     'genesee_fuel_&_heating_co,_inc' 'u_of_w_building_permit'
     'home_run_electric_llc' 'raymark_plumbing_&_sewer' 'techpros_llc'
     "rob's_electric_incorporated" 'northwest_electric_&_solar_llc'
     'meridian_center_electric_inc' 'seattle_school_district_(a&s)'
     'endeavor_telecom_inc' 'boeing_commercial_airplane_grp' 'owner'
     'burgess_design' 'playhouse_design_group' 'blueprint_services,_llc'
     'julian_weber_arch_&_design_llc' 'sme_inc_of_seattle' 'novion_group_inc'
     'wesco' 'jim_dandy_sewer_&_plumbing' 'evergreen_power_systems,_inc']
    Uniques:  101


# Legal Description


```python
print("Count of Uniques: ", len(X_train['legaldescription'].unique()))
print("Uniques: ", X_train['legaldescription'].unique())

```

    Count of Uniques:  14388
     [nan
     'LOT 3 TOGETHER WITH THE NORTH HALF OF LOT 4, BLOCK 17, LAWS 2ND ADDN'
     'MARKET PLACE QUEEN ANNE CONDO, VOL 17, PAGE 8' ...
     'PARCEL A, LBA #3005752'
     "SLY 30' OF LOT 16, ALL OF LOTS 17 & 18 & NE 5.2' OF LOT 19, BLK 12, EAST PARK ADDITION"
     'W 53 FT OF S 1/2, TR 2, BARTOS ACRE TRACTS']


Looking at the legal descriptions, we can make out that they are not very informative when it comes to making predictions. So we might just turn it into a boolean `has_LD`.


```python
X_train.loc[:,'legaldescription'] = X_train['legaldescription'].fillna('None')
X_train.loc[:,'has_ld'] = [1 if i!='None' else 0 for i in X_train['legaldescription']]

```

# Description


```python
print("Count of Uniques: ", len(X_train['description'].unique()))
print("Uniques: ", X_train['description'].unique()[:100])

```

    Count of Uniques:  80143
    Uniques:  ['INSTALL LOW VOLTAGE SECURITY SYSTEM'
     'INSTALL (1) NON-ILLUMINATED WALL SIGN - PLAZA GARIBALDI'
     'INSTALL (1) NON-ILLUM ON-PREMISES WALL SIGN FOR &quot;PLAZA GARIBALDI&quot;'
     'REPLACE OIL FURNACE W/ NEW GAS FURNACE'
     'WIRE NEW SINGLE FAMILY RESIDENCE W/ 200 AMP SERVICE' 'REPAIR'
     'Construct addition and alterations to existing single family residence, per plan.'
     'ADDING CELLGUARD' 'Rebuild Letter' 'CAB INTERIOR UPGRADES'
     '[TEMPORARY POWER. ]' 'Cancel per customer log 11-469'
     'Expansion of existing minor communication facility consisting of removal and replacement of antennas and rooftop screening per plan.'
     'Demo existing single family dwelling.' 'PREWIRE SECURITY SYSTEM'
     'INSTALL TWO NEW 15 AMP CIRCUITS. REPLACE UNGROUNDED WIRING IN CRAWLSPACE.'
     'NEW 200 AMP SERVICE' 'INSTALL HEAT PUMP TO SERVE BEDROOM & ATTIC'
     'REROUTE SEWER & REPLACE DOWNSPOUT' 'Cancel per customer log 11-177'
     'INSTALLING (2) DUCT SMOKE DETECTORS W/AUDIBLE TEST STATIONS, (2) T-STATS W/RETURN AIR DUCT SENSORS'
     'F/A TI'
     'Change of use from general retail sales & service to Medical office and construct interior, non-structural tenant improvements to existing mixed use building on the 2nd floor/occupy per plan.'
     'WIRING FOR TI; POWER & LIGHTS. APE'
     'Remove existing detached garage, established parking to remain, subject to field inspection.'
     'REPLACE EXISTING BOILER' 'KITCHEN LIGHTING AND LIVINGROOM OUTLET'
     'SOLAR POWERED PARKING PAY & DISPLAY UNIT.'
     'AC POWERED PARKING PAY & DISPLAY UNIT.'
     'Land Use Application to subdivide one parcel into two parcels of land. Proposed parcel sizes are: A) 3,645 sq. ft. and B) 1,800 sq. ft. Existing structures on proposed Parcel A and B to remain.'
     'Construct 1-story addition to west side of existing single family residence per plan.'
     'Construct deck to serve adjacent restaurant, per plan.'
     'CLOSED AS INCOMPLETE - EXPIRED PERMIT. REPAIR EXISTING SANITARY SEWER & INSTALL PIPING FOR ADDITIONAL DOWNSPOUTS'
     '(1) FURNACE CHANGE OUT & (1) NEW FURNACE IN SFR'
     'RELOCATE METER BASE, WIRE REMODEL WITH ADDITIONS ADDED, AND NEW FURNACE. 9/30/09 BEW ADDED 200 AMP SERVICE'
     'Establish an ADU in an existing single family residence per plan'
     'INSTALL TEMP POWER POLE FOR CONSTRUCTION 1@001-125 AMP'
     'INSTALL NEW GAS FURNACE'
     'INSTALL ELECTRICAL WIRING FOR SECOND STORY REMODEL, INSTALL NEW SERVICE METER'
     'Construct improvements to basement of single family residence per plan and subject to field inspection (STFI).'
     '200 AMP SERVICE UPGRADE WITH SQUARE D QO 200 AMP PANEL. WIRE BASEMENT.'
     'TEMP POWER POLE - RFR BDLG A/P #6179213' 'SIDE SEWER REPAIR'
     'Construct full 2nd floor addition to existing songe family residence per plan.'
     'ENTIRE HOUSE REMODEL, REWIRING ENTIRE ELECTRICAL SYSTEM'
     'Construct dormer additions east side, enclose porch on NE corner, and alter portion of basement. Remove existing garage and provide off street parking, per floor plans and subject to field inspection, STFI.'
     'WIRE KITCHEN REMODEL & SERVICE CHANGE. 5/6/08 JNL - ABOUT TO EXPIRE LETTER SENT. 5/26/09 BEW RENEWED'
     'RECOVER EXISTING AWNING W/ GRAPHICS & INSTALL (3) ILLUM WALL SIGNS - VERIZON WIRELESS'
     'UFER GROUND - REF #6118741' 'FIRE ALARM T.I.'
     'Closed as Incomplete - Expired Permit Interior alterations to whole house, enclose back porch, no structural work, finish off basement to create bedrooms/bathrooms per plot plan and stfi'
     'Construct addition to existing detached accessory garage per plot plan and stfi (revised 1/23/07)'
     '200 AMP SERVICE UPGRADE. REPLACE EXISTING ZINSCO PANEL IN OUTBUILDING. INSTALL RECEPTACLES FOR COMPUTERS.'
     'KITCHEN REMODEL REF: 6104340' 'KITCHEN, DINING ROOM LIVING ROOM REMODEL'
     'WIRING FOR KITCHEN REMODEL'
     'REPLACE RUBBER FLR W/NEW FOR KEY ARENA FREIGHT ELEVATOR'
     'REPLACE GAS FURNACE IN SFR'
     'REPLACE EXISITNG CIRCUIT. NEW DRYER CIRCUIT. REPLACE TWO EXISTING CIRCUITS'
     'REMODEL AND UPDATE ELECTRICAL, SWITCH, OUTLET & LIGHTING'
     'Closed as Incomplete - Expired Permit UNDERPIN 20 FEET OF NORTH FOUNDATION WALL, PER PLAN'
     'SEWER REPAIR'
     'ERECT TEMPORARY TENT OVER OUTDOOR SEATING AREA OF EXIST. RESTAURANT 11/27/04 THRU 5/27/05 PER PLANS LESSEE RESPONSIBLE FOR TENT STRUCTURAL INTEGRITY'
     'WIRE ADDITION 2 BED ROOM AND 1 BATH.'
     'UPGRADE TO A FULL 200 AMP SERVICE AND WIRE NEW ADDITION'
     'SERVICE CHANGE 200 AMP. SFR' 'REWIRE TO KITCHEN REMODEL PROJECT. SFR'
     'INSTALL NEW TYPE I COMMERCIAL HOOD IN FAST FOOD RESTAURANT PER PLAN'
     'RENOVATION OF EXISTING RESTAURANT/ NEW SEATING/ ADA BTHMS/ PERFORMANCE PLATFORM/ NON-STRUCTURAL WALLS/ OCCUPY PER PLANS'
     'CONSTRUCT ADDITION TO SECOND FLOOR OF SINGLE FAMILY RESIDENCE, PER PLAN'
     'CONSTRUCT DORMER ADDITION TO EXISTING SINGLE FAMILY RESIDENCE ALL PER PLAN & SUBJECT TO FIELD INSPECTION.'
     'SET UP FENCE AROUND PARKING AREA, ERECT CANOPIES AND LIGHTS FOR BEER GARDEN MARCH 2 - MARCH 7, 2000 PER PLOT PLAN'
     'ESTABLISH TEMPORARY USE FOR BEER GARDEN FOR 30 DAY FROM 4/30/99 ENDING 5/30/99. PER PLOT PLAN SUBJECT TO FIELD INSPECTION.'
     'ADD 6 FOOT RETAINING WALL IN THE REAR YARD TO REPLACE EXISTING FAILING RETAINING WALL'
     'CONSTRUCT GROUND LEVEL ADDITION AND DECK ADDITION TO A SINGLE FAMILY RESIDENCE, PER PLOT PLAN & SUBJECT TO FIELD INSPECTION, STFI'
     'DEMO EXISTING SINGLE FAMILY RESIDENCE ESTABLISH USE AS SINGLE FAMILY RESIDENCE CONSTRUCT ONE FAMILY DWELLING WITH ATTACHED GARAGE PER PLANS'
     'INTERIOR ALTERATIONS DIVIDE 1 RETAIL SPACE INTO 2 SPACES PER PLAN'
     'REMODEL SECOND FLOOR OF A SINGLE FAMILY RESIDENCE SUBJECT TO FIELD INSPECTION (STFI)'
     'INTERIOR ALTERATIONS (KITCHEN REMODEL) TO SINGLE FAMILY RESIDENCE SUBJECT TO FIELD INSPECTION'
     'REPLACE TWO (2) ROOFTOP UNITS IN KIND, SUBJECT TO FIELD INSPECTION (STFI)'
     'ALTERATIONS TO LEVEL REAR OF EXISTING SFR & TO FINISH PORTION OF BASEMENT LEVEL FOR REC. ROOM SUBJECT TO FIELD INSPECTION (STFI)'
     'CONSTRUCT AN EXTERIOR CONVENIENCE STAIRWAY, PER PLANS'
     'ESTABLISH USE FOR RECORDS AS TWO SINGLE FAMILY RESIDENCES ON ONE SITE, PER PLANS'
     'INSTALL HVAC IN EXISTING TAVERN PER PLAN'
     'GRADE 2000 CU. YDS. OFF MATERIAL AND INSTALL SOIL NAILING SYSTEM/SHORING, PER PLANS'
     'CONSTRUCT 1ST FLR BAY WINDOW ADDITION AND ALTERATIONS TO KITCHEN OF SINGLE FAMILY REISDENCE PER PLANS'
     'ADD ONTO KITCHEN AND ADD DECK ALTER KITCHEN ALL PER PLANS'
     'ADD DORMER TO EXISTING SFR,PER PLANS'
     'KITCHEN & DECK ADDITION PER PLOT PLAN & STFI'
     'RE-STRIPE PARKING LOT, PER PLANS'
     'TO ESTABLISH SIDE WALK CAFE FOR THAI RESTAURANT'
     'INSTALL (1) 7-DAY PROGRAMMABLE THERMOSTAT AND LOW VOLTAGE WIRING'
     'Mechanical STFI - Adding 1 exhaust fan, 2 openings and 1 dryer vent to SPS McClure Middle School to floor 1 per plans.'
     'Replace six heat pumps with new heat pumps'
     '(PRIME JOB# 1139388) ADD SUB PANEL ONLY AND EMON DEMON METERING'
     'INSTALL CLEAN OUT' 'Revision to decrease size of deck'
     'REMODEL IN CONDO #302' 'REPLACING BAD FIRE ALARM CONTROL PANEL'
     'Alterations to minor communication utility to replace cabinets and associated equipment on rooftop of existing office building, per plan']


Looking at some of the descriptions we can immediately make out that some of the terms may be specific to ELECTRICAL type and we might want to identify a way to account for these terms. We will be suing NLTK to compute the TF-IDF for these descriptions.


```python
# Remove stopwords, non alphabetic characters


def nltk_description(data):

    # Split data to analyze patterns in Electrical type vs non
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    num_pattern = re.compile(r'\s*[\W0-9\s]\s*')

    # Fill NaN in licensetype
    data.loc[:,'description'] = data.loc[:,'description'].fillna('None')

    sample = []

    for index, description in enumerate(data["description"]):

        words = word_tokenize(description)
        no_stops = [i for i in words if i.lower() not in stop_words]
        no_special_char = [ps.stem(num_pattern.sub("",i)) for i in no_stops if ps.stem(num_pattern.sub("",i)) != '']
        descrip = " ".join(i for i in no_special_char)
        sample.append(descrip)

    cv = TfidfVectorizer(min_df=1, stop_words='english')
    x_traincv = cv.fit_transform(sample)

    return x_traincv.toarray().sum(axis=1).reshape(data.shape[0])

```


```python
X_train.loc[:,'tf-idf_description'] = nltk_description(X_train)
print(X_train['tf-idf_description'])
```

    0         1.944464
    1         2.240490
    2         2.620758
    3         2.083272
    4         2.602990
                ...   
    100151    2.219222
    100152    2.822032
    100153    3.397444
    100154    2.879694
    100155    1.727000
    Name: tf-idf_description, Length: 100156, dtype: float64


# Type


```python
print("Count of Uniques: ", len(X_train['type'].unique()))
print("Uniques: ", X_train['type'].unique())

```

# Subtype


```python
print("Count of Uniques: ", len(X_train['subtype'].unique()))
print("Uniques: ", X_train['subtype'].unique())

```

    Count of Uniques:  6
    Uniques:  ['COMMERCIAL' 'SINGLE FAMILY / DUPLEX' nan 'MULTIFAMILY' 'INDUSTRIAL'
     'INSTITUTIONAL']



```python
X_train['subtype'] = X_train['subtype'].fillna('None')
[i.lower().replace(" ", "_") for i in X_train['subtype']]
```




    ['commercial',
     'commercial',
     'commercial',
     'single_family_/_duplex',
     'single_family_/_duplex',
     'single_family_/_duplex',
     'single_family_/_duplex',
     'single_family_/_duplex',
     'none',
     'none',
     'single_family_/_duplex',
     ...
     ...
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     'none',
     'none',
     'commercial',
     'commercial',
     'commercial',
     'commercial',
     ...]



# Job value


```python
print("Count of Uniques: ", len(X_train['job_value'].unique()))
print("Uniques: ", X_train['job_value'].unique())

```

    Count of Uniques:  5132
    Uniques:  [nan '$250,000.00' '$45,000.00' ... '$321,743.00' '$164,848.00'
     '$732,408.00']



```python
cleaned_job_value = X_train['job_value'].apply(lambda x: float(str(x).replace('$', '').replace(',','')))
X_train.loc[:,'job_value'] = cleaned_job_value
X_train.loc[:,'job_value'] = X_train['job_value'].fillna(0.0)
X_train.head(200)

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
      <th>licensetype</th>
      <th>businessname</th>
      <th>legaldescription</th>
      <th>description</th>
      <th>type</th>
      <th>subtype</th>
      <th>job_value</th>
      <th>has_ld</th>
      <th>tf-idf_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>adt_llc</td>
      <td>None</td>
      <td>INSTALL LOW VOLTAGE SECURITY SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.944464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL (1) NON-ILLUMINATED WALL SIGN - PLAZA ...</td>
      <td>SIGN/BILLBOARD</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.240490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SPECIALTY CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL (1) NON-ILLUM ON-PREMISES WALL SIGN FO...</td>
      <td>SIGN/BILLBOARD</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.620758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONT...</td>
      <td>olson_energy_service</td>
      <td>None</td>
      <td>REPLACE OIL FURNACE W/ NEW GAS FURNACE</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.083272</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>WIRE NEW SINGLE FAMILY RESIDENCE W/ 200 AMP SE...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.602990</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>metropolitan_sewer_service_llc</td>
      <td>None</td>
      <td>REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>none</td>
      <td>LOT 3 TOGETHER WITH THE NORTH HALF OF LOT 4, B...</td>
      <td>Construct addition and alterations to existing...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>250000.0</td>
      <td>1</td>
      <td>2.799645</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>adt_llc</td>
      <td>None</td>
      <td>ADDING CELLGUARD</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.377901</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>none</td>
      <td>MARKET PLACE QUEEN ANNE CONDO, VOL 17, PAGE 8</td>
      <td>Rebuild Letter</td>
      <td>LU POLICY</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.414076</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>thyssenkrupp_elevator_corp</td>
      <td>None</td>
      <td>CAB INTERIOR UPGRADES</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.669762</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>[TEMPORARY POWER. ]</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.410228</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>Cancel per customer log 11-469</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.729264</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NaN</td>
      <td>none</td>
      <td>198820-0055; S 110' OF E 120', BLOCK 24, D T D...</td>
      <td>Expansion of existing minor communication faci...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>45000.0</td>
      <td>1</td>
      <td>3.301165</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Demo existing single family dwelling.</td>
      <td>DEMOLITION / DEMOLITION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.129480</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>PREWIRE SECURITY SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.378026</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL TWO NEW 15 AMP CIRCUITS. REPLACE UNGRO...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.517140</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>NEW 200 AMP SERVICE</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.702799</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL HEAT PUMP TO SERVE BEDROOM &amp; ATTIC</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.330450</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>REROUTE SEWER &amp; REPLACE DOWNSPOUT</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.908974</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NaN</td>
      <td>none</td>
      <td>LOTS 11 &amp; 12, BLK 48, LAWS 2ND ADDITION AKA: F...</td>
      <td>Cancel per customer log 11-177</td>
      <td>LAND USE</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.729264</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>merit_mechanical_inc</td>
      <td>None</td>
      <td>INSTALLING (2) DUCT SMOKE DETECTORS W/AUDIBLE ...</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.108278</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>F/A TI</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.412895</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NaN</td>
      <td>none</td>
      <td>MARKET PLACE QUEEN ANNE CONDO, VOL 17, PAGE 8</td>
      <td>Change of use from general retail sales &amp; serv...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>170000.0</td>
      <td>1</td>
      <td>4.084502</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRING FOR TI; POWER &amp; LIGHTS. APE</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.139505</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Remove existing detached garage, established p...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>1000.0</td>
      <td>0</td>
      <td>3.070862</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GENERAL CONTRACTOR LICENSE, REFRIGERATION CONT...</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACE EXISTING BOILER</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.640491</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>boyer_electric_company_inc</td>
      <td>None</td>
      <td>KITCHEN LIGHTING AND LIVINGROOM OUTLET</td>
      <td>ELECTRICAL</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.884571</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>SOLAR POWERED PARKING PAY &amp; DISPLAY UNIT.</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.326732</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>AC POWERED PARKING PAY &amp; DISPLAY UNIT.</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.321862</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>none</td>
      <td>PARCAL A, SP #3010940</td>
      <td>Land Use Application to subdivide one parcel i...</td>
      <td>LAND USE</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.964933</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NaN</td>
      <td>none</td>
      <td>E 80' OF S 10' OF LT 3, ALL OF LT 4, BLK 48, L...</td>
      <td>Construct 1-story addition to west side of exi...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>60000.0</td>
      <td>1</td>
      <td>2.866415</td>
    </tr>
    <tr>
      <th>31</th>
      <td>NaN</td>
      <td>none</td>
      <td>LOT 12, BLOCK 33, D.T. DENNY'S HOME ADDN</td>
      <td>Construct deck to serve adjacent restaurant, p...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>20558.0</td>
      <td>1</td>
      <td>2.271890</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>CLOSED AS INCOMPLETE - EXPIRED PERMIT. REPAIR ...</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.311431</td>
    </tr>
    <tr>
      <th>33</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION C...</td>
      <td>Other</td>
      <td>None</td>
      <td>(1) FURNACE CHANGE OUT &amp; (1) NEW FURNACE IN SFR</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.839842</td>
    </tr>
    <tr>
      <th>34</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>RELOCATE METER BASE, WIRE REMODEL WITH ADDITIO...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.256178</td>
    </tr>
    <tr>
      <th>35</th>
      <td>NaN</td>
      <td>none</td>
      <td>LOTS 104-105, DWIGHTS ADDN</td>
      <td>Establish an ADU in an existing single family ...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>8000.0</td>
      <td>1</td>
      <td>2.434005</td>
    </tr>
    <tr>
      <th>36</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL TEMP POWER POLE FOR CONSTRUCTION 1@001...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.339861</td>
    </tr>
    <tr>
      <th>37</th>
      <td>GENERAL CONTRACTOR LICENSE, REFRIGERATION CONT...</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL NEW GAS FURNACE</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.943749</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL ELECTRICAL WIRING FOR SECOND STORY REM...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.937852</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Construct improvements to basement of single f...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>4000.0</td>
      <td>0</td>
      <td>3.249149</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>200 AMP SERVICE UPGRADE WITH SQUARE D QO 200 A...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.683281</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>TEMP POWER POLE - RFR BDLG A/P #6179213</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.316230</td>
    </tr>
    <tr>
      <th>42</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>metropolitan_sewer_service_llc</td>
      <td>None</td>
      <td>SIDE SEWER REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.413594</td>
    </tr>
    <tr>
      <th>43</th>
      <td>GENERAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>LOT 3, BLOCK 48, BIGELOWS ADD</td>
      <td>Construct full 2nd floor addition to existing ...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>119300.0</td>
      <td>1</td>
      <td>2.493348</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>ENTIRE HOUSE REMODEL, REWIRING ENTIRE ELECTRIC...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.915969</td>
    </tr>
    <tr>
      <th>45</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Construct dormer additions east side, enclose ...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>190000.0</td>
      <td>0</td>
      <td>4.574784</td>
    </tr>
    <tr>
      <th>46</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRE KITCHEN REMODEL &amp; SERVICE CHANGE. 5/6/08 ...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.220609</td>
    </tr>
    <tr>
      <th>47</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, GENERAL CONTRAC...</td>
      <td>Other</td>
      <td>None</td>
      <td>RECOVER EXISTING AWNING W/ GRAPHICS &amp; INSTALL ...</td>
      <td>SIGN/BILLBOARD</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.967156</td>
    </tr>
    <tr>
      <th>48</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>UFER GROUND - REF #6118741</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.727664</td>
    </tr>
    <tr>
      <th>49</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>FIRE ALARM T.I.</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.411078</td>
    </tr>
    <tr>
      <th>50</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Closed as Incomplete - Expired Permit Interior...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>55000.0</td>
      <td>0</td>
      <td>3.940076</td>
    </tr>
    <tr>
      <th>51</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Construct addition to existing detached access...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>3777.0</td>
      <td>0</td>
      <td>3.064270</td>
    </tr>
    <tr>
      <th>52</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>200 AMP SERVICE UPGRADE. REPLACE EXISTING ZINS...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.007737</td>
    </tr>
    <tr>
      <th>53</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>KITCHEN REMODEL REF: 6104340</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.718972</td>
    </tr>
    <tr>
      <th>54</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>KITCHEN, DINING ROOM LIVING ROOM REMODEL</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.155358</td>
    </tr>
    <tr>
      <th>55</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>WIRING FOR KITCHEN REMODEL</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.728293</td>
    </tr>
    <tr>
      <th>56</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>kone_inc</td>
      <td>None</td>
      <td>REPLACE RUBBER FLR W/NEW FOR KEY ARENA FREIGHT...</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.704942</td>
    </tr>
    <tr>
      <th>57</th>
      <td>REFRIGERATION CONTRACTOR LIC., ELECTRICAL CONT...</td>
      <td>blue_flame_llc</td>
      <td>None</td>
      <td>REPLACE GAS FURNACE IN SFR</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.995563</td>
    </tr>
    <tr>
      <th>58</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACE EXISITNG CIRCUIT. NEW DRYER CIRCUIT. R...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.163233</td>
    </tr>
    <tr>
      <th>59</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REMODEL AND UPDATE ELECTRICAL, SWITCH, OUTLET ...</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.412186</td>
    </tr>
    <tr>
      <th>60</th>
      <td>NaN</td>
      <td>none</td>
      <td>LT 1, &amp; BLK 9, PLEASANT RIDGE ADDITION &amp; N 24'...</td>
      <td>Closed as Incomplete - Expired Permit UNDERPIN...</td>
      <td>PTS PERMIT</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.002104</td>
    </tr>
    <tr>
      <th>61</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>metropolitan_sewer_service_llc</td>
      <td>None</td>
      <td>SEWER REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.413594</td>
    </tr>
    <tr>
      <th>62</th>
      <td>NaN</td>
      <td>guardian_security_systems_inc</td>
      <td>None</td>
      <td>INSTALL LOW VOLTAGE SECURITY SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.944464</td>
    </tr>
    <tr>
      <th>63</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ERECT TEMPORARY TENT OVER OUTDOOR SEATING AREA...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.195211</td>
    </tr>
    <tr>
      <th>64</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRE ADDITION 2 BED ROOM AND 1 BATH.</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.116130</td>
    </tr>
    <tr>
      <th>65</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>UPGRADE TO A FULL 200 AMP SERVICE AND WIRE NEW...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.407681</td>
    </tr>
    <tr>
      <th>66</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>SERVICE CHANGE 200 AMP. SFR</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.999897</td>
    </tr>
    <tr>
      <th>67</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REWIRE TO KITCHEN REMODEL PROJECT. SFR</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.225021</td>
    </tr>
    <tr>
      <th>68</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL NEW TYPE I COMMERCIAL HOOD IN FAST FOO...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.789960</td>
    </tr>
    <tr>
      <th>69</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>RENOVATION OF EXISTING RESTAURANT/ NEW SEATING...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.283908</td>
    </tr>
    <tr>
      <th>70</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT ADDITION TO SECOND FLOOR OF SINGLE F...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.699026</td>
    </tr>
    <tr>
      <th>71</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT DORMER ADDITION TO EXISTING SINGLE F...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.173906</td>
    </tr>
    <tr>
      <th>72</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>SET UP FENCE AROUND PARKING AREA, ERECT CANOPI...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.990347</td>
    </tr>
    <tr>
      <th>73</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ESTABLISH TEMPORARY USE FOR BEER GARDEN FOR 30...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.160716</td>
    </tr>
    <tr>
      <th>74</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ADD 6 FOOT RETAINING WALL IN THE REAR YARD TO ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.754667</td>
    </tr>
    <tr>
      <th>75</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT GROUND LEVEL ADDITION AND DECK ADDIT...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.558921</td>
    </tr>
    <tr>
      <th>76</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>DEMO EXISTING SINGLE FAMILY RESIDENCE ESTABLIS...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.207132</td>
    </tr>
    <tr>
      <th>77</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INTERIOR ALTERATIONS DIVIDE 1 RETAIL SPACE INT...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.196264</td>
    </tr>
    <tr>
      <th>78</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>REMODEL SECOND FLOOR OF A SINGLE FAMILY RESIDE...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.077607</td>
    </tr>
    <tr>
      <th>79</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INTERIOR ALTERATIONS (KITCHEN REMODEL) TO SING...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.117723</td>
    </tr>
    <tr>
      <th>80</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>REPLACE TWO (2) ROOFTOP UNITS IN KIND, SUBJECT...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.703180</td>
    </tr>
    <tr>
      <th>81</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ALTERATIONS TO LEVEL REAR OF EXISTING SFR &amp; TO...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.455951</td>
    </tr>
    <tr>
      <th>82</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT AN EXTERIOR CONVENIENCE STAIRWAY, PE...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.033160</td>
    </tr>
    <tr>
      <th>83</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ESTABLISH USE FOR RECORDS AS TWO SINGLE FAMILY...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.649213</td>
    </tr>
    <tr>
      <th>84</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL HVAC IN EXISTING TAVERN PER PLAN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.922489</td>
    </tr>
    <tr>
      <th>85</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>GRADE 2000 CU. YDS. OFF MATERIAL AND INSTALL S...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.745705</td>
    </tr>
    <tr>
      <th>86</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT 1ST FLR BAY WINDOW ADDITION AND ALTE...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.047379</td>
    </tr>
    <tr>
      <th>87</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ADD ONTO KITCHEN AND ADD DECK ALTER KITCHEN AL...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.995064</td>
    </tr>
    <tr>
      <th>88</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ADD DORMER TO EXISTING SFR,PER PLANS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.091918</td>
    </tr>
    <tr>
      <th>89</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>KITCHEN &amp; DECK ADDITION PER PLOT PLAN &amp; STFI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.384654</td>
    </tr>
    <tr>
      <th>90</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>RE-STRIPE PARKING LOT, PER PLANS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.815897</td>
    </tr>
    <tr>
      <th>91</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>TO ESTABLISH SIDE WALK CAFE FOR THAI RESTAURANT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.137136</td>
    </tr>
    <tr>
      <th>92</th>
      <td>GENERAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL (1) 7-DAY PROGRAMMABLE THERMOSTAT AND ...</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.470726</td>
    </tr>
    <tr>
      <th>94</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>Mechanical STFI - Adding 1 exhaust fan, 2 open...</td>
      <td>MECHANICAL / ALTER EXISTING INSTALLATION</td>
      <td>INDUSTRIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.503225</td>
    </tr>
    <tr>
      <th>95</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Replace six heat pumps with new heat pumps</td>
      <td>MECHANICAL / NEW INSTALLATION</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.805701</td>
    </tr>
    <tr>
      <th>96</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>prime_electric_incorporated</td>
      <td>None</td>
      <td>(PRIME JOB# 1139388) ADD SUB PANEL ONLY AND EM...</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.665892</td>
    </tr>
    <tr>
      <th>97</th>
      <td>GENERAL CONTRACTOR LICENSE, GENERAL CONTRACTOR...</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL CLEAN OUT</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.274943</td>
    </tr>
    <tr>
      <th>98</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Revision to decrease size of deck</td>
      <td>POST ISSUANCE SUBMITTAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.886059</td>
    </tr>
    <tr>
      <th>99</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>active_engineering</td>
      <td>None</td>
      <td>REMODEL IN CONDO #302</td>
      <td>ELECTRICAL</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.378499</td>
    </tr>
    <tr>
      <th>100</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACING BAD FIRE ALARM CONTROL PANEL</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.066037</td>
    </tr>
    <tr>
      <th>101</th>
      <td>NaN</td>
      <td>Other</td>
      <td>LTS 7 THRU 9, BLK 20, D.T.DENNY'S NORTH SEATTL...</td>
      <td>Alterations to minor communication utility to ...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>20000.0</td>
      <td>1</td>
      <td>3.449173</td>
    </tr>
    <tr>
      <th>102</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>Closed without final inspection, no known life...</td>
      <td>ELECTRICAL</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.941935</td>
    </tr>
    <tr>
      <th>103</th>
      <td>GENERAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>Post Permit Sprinklers</td>
      <td>POST ISSUANCE SUBMITTAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.680537</td>
    </tr>
    <tr>
      <th>104</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONT...</td>
      <td>olson_energy_service</td>
      <td>None</td>
      <td>REPLACE OIL FURNACE W/ NEW GAS FURNACE</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.083272</td>
    </tr>
    <tr>
      <th>105</th>
      <td>GENERAL CONTRACTOR LICENSE, REFRIGERATION CONT...</td>
      <td>Other</td>
      <td>None</td>
      <td>BOILER REPLACEMENT</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.380524</td>
    </tr>
    <tr>
      <th>106</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>thyssenkrupp_elevator_corp</td>
      <td>None</td>
      <td>CONTROLLER, MACHINE, SHEAVES, GOVERNOR, CAR ST...</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.730349</td>
    </tr>
    <tr>
      <th>107</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Connect 3 new supply grilles to existing air h...</td>
      <td>MECHANICAL / ALTER EXISTING INSTALLATION</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.709357</td>
    </tr>
    <tr>
      <th>108</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Construct interior alterations to existing off...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>136000.0</td>
      <td>0</td>
      <td>3.286962</td>
    </tr>
    <tr>
      <th>109</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>ADD CIRCUIT FOR SISTERN SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.526010</td>
    </tr>
    <tr>
      <th>110</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACE KNOB AND TUBE IN ATTIC SPACE. UPGRADE ...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.554674</td>
    </tr>
    <tr>
      <th>111</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>RELOCATING (5) T-STATS</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.398731</td>
    </tr>
    <tr>
      <th>112</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, GENERAL CONTRAC...</td>
      <td>reed_wright_htg_&amp;_elec_co_inc</td>
      <td>None</td>
      <td>REPLACE EXISTING PORTABLE HOT WATER BOILER WIT...</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.716072</td>
    </tr>
    <tr>
      <th>113</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>Other</td>
      <td>None</td>
      <td>SEWER REPAIR IN STREET</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.692555</td>
    </tr>
    <tr>
      <th>114</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRE 2ND FLR ADDITION, INSTALL 70 AMP FEEDER</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.564075</td>
    </tr>
    <tr>
      <th>115</th>
      <td>NaN</td>
      <td>none</td>
      <td>515 First Avenue West (Condo), Vol 144 pgs 90-94</td>
      <td>Alter office unit into 2 offices unit of 1st f...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>20000.0</td>
      <td>1</td>
      <td>2.210897</td>
    </tr>
    <tr>
      <th>116</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>active_engineering</td>
      <td>None</td>
      <td>ADD CIRCUITS FOR ROOFTOP CHILLER TO SERVE LIVI...</td>
      <td>ELECTRICAL</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.740991</td>
    </tr>
    <tr>
      <th>117</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>Relocate 35 diffusers and ductwork per plot plan</td>
      <td>MECHANICAL / ALTER EXISTING INSTALLATION</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.163306</td>
    </tr>
    <tr>
      <th>118</th>
      <td>NaN</td>
      <td>Other</td>
      <td>LOTS 1 THRU 6, BLK 15, D.T. DENNY'S NO. SEATTL...</td>
      <td>Blanket Permit for non-structural tenant impro...</td>
      <td>BLANKET</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.675731</td>
    </tr>
    <tr>
      <th>119</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>Other</td>
      <td>None</td>
      <td>REPAIR SEWER LINE. 8/26/09 BEW ADDED WORK IN ROW</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.601298</td>
    </tr>
    <tr>
      <th>120</th>
      <td>NaN</td>
      <td>none</td>
      <td>THE N 28 FT OF LOT 15 &amp; THE S 5 FT OF LOT 14, ...</td>
      <td>Revision to revise scope of the top floor remo...</td>
      <td>POST ISSUANCE SUBMITTAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.014014</td>
    </tr>
    <tr>
      <th>121</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>RELOCATING (4) F/A SPEAKERS</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.712711</td>
    </tr>
    <tr>
      <th>122</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REWIRE OF EXISTING SFR</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.654067</td>
    </tr>
    <tr>
      <th>123</th>
      <td>NaN</td>
      <td>none</td>
      <td>LTS 6-8, 14-16, BLK 11, COVE ADDNTGW ALL OF BL...</td>
      <td>CLOSED AS INCOMPLETE - EXPIRED PERMIT. Constru...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>20000.0</td>
      <td>1</td>
      <td>3.558370</td>
    </tr>
    <tr>
      <th>124</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION C...</td>
      <td>merit_mechanical_inc</td>
      <td>None</td>
      <td>REPLACE EXISTING BOILER WITH NEW</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.889408</td>
    </tr>
    <tr>
      <th>125</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>UPGRADE ELECTRICAL AND FA SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>INSTITUTIONAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.725983</td>
    </tr>
    <tr>
      <th>126</th>
      <td>REFRIGERATION CONTRACTOR LIC., GENERAL CONTRAC...</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACE EXISTING GAS FURNACE WITH NEW 93%</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.186389</td>
    </tr>
    <tr>
      <th>127</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>UPDATE 125 AMP SERVICE BY REPLACING PANEL BOX,...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.742985</td>
    </tr>
    <tr>
      <th>128</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRE NEW GARAGE TO REPLACE EXISTING. REPLACE O...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.633916</td>
    </tr>
    <tr>
      <th>129</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Remove existing detached rear garage and const...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>23339.0</td>
      <td>0</td>
      <td>3.614723</td>
    </tr>
    <tr>
      <th>130</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALLING AUDIO/VIDEO, LIGHTING, SECURITY AND...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.222622</td>
    </tr>
    <tr>
      <th>131</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>roto_rooter_services_company</td>
      <td>None</td>
      <td>SIDE SEWER REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.413594</td>
    </tr>
    <tr>
      <th>132</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>titan_electric_inc</td>
      <td>None</td>
      <td>GET STARTED PERMIT FOR FIRE ALARM SYSTEM. REF ...</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.346189</td>
    </tr>
    <tr>
      <th>133</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Alterations to existing (basement and stairs o...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>50000.0</td>
      <td>0</td>
      <td>3.500738</td>
    </tr>
    <tr>
      <th>134</th>
      <td>NaN</td>
      <td>none</td>
      <td>LOTS 1 THRU 9, BLOCK 18, COMSTOCK ADD</td>
      <td>Alter interior, new south entrance and skyligh...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>COMMERCIAL</td>
      <td>600000.0</td>
      <td>1</td>
      <td>2.716049</td>
    </tr>
    <tr>
      <th>135</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>Closed without final inspection, no known life...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.685112</td>
    </tr>
    <tr>
      <th>136</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>valley_elec_co_of_mtvernon_inc</td>
      <td>None</td>
      <td>INSTALLATION OF LOW VOLTAGE CONTROLS.</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.939860</td>
    </tr>
    <tr>
      <th>137</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION C...</td>
      <td>evergreen_refrigeration_llc</td>
      <td>None</td>
      <td>Relocate (5) existing diffusers, and minor duc...</td>
      <td>MECHANICAL / ALTER EXISTING INSTALLATION</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.386916</td>
    </tr>
    <tr>
      <th>138</th>
      <td>REFRIGERATION CONTRACTOR LIC., GENERAL CONTRAC...</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACING BOILER B-1</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.380524</td>
    </tr>
    <tr>
      <th>139</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, ELECTRICAL CONT...</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL HVAC/REFER CONTROL SYSTEM</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.483593</td>
    </tr>
    <tr>
      <th>140</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Demolish existing single-family dwelling per p...</td>
      <td>DEMOLITION /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.019120</td>
    </tr>
    <tr>
      <th>141</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ADD 5 CAN LIGHTS FOR EXTERIOR PORCH; RELOCATE ...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.688026</td>
    </tr>
    <tr>
      <th>142</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>o'neill_plumbing_co</td>
      <td>None</td>
      <td>INSTALLING 6 INCH C/O IN FRONT YARD</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.605453</td>
    </tr>
    <tr>
      <th>143</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>Porch additions to single family residence per...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>17976.0</td>
      <td>0</td>
      <td>2.319121</td>
    </tr>
    <tr>
      <th>144</th>
      <td>REGIS. SIDE SEWER CONTRACTOR</td>
      <td>Other</td>
      <td>None</td>
      <td>NEW SEWER &amp; DRAINAGE</td>
      <td>SIDE SEWER</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.659887</td>
    </tr>
    <tr>
      <th>145</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>metropolitan_sewer_service_llc</td>
      <td>None</td>
      <td>REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>146</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, GENERAL CONTRAC...</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL NON-ILLUM MONUMENT ID SIGN - ST ANNE C...</td>
      <td>SIGN/BILLBOARD</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.838452</td>
    </tr>
    <tr>
      <th>147</th>
      <td>NaN</td>
      <td>none</td>
      <td>CARRARA II CONDOS, VOL 170, PGS. 13-19</td>
      <td>REMOVE, REPAIR AND REPLACE DAMAGED SIDING, SHE...</td>
      <td>CONSTRUCTION / ADDITION OR ALTERATION</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.893337</td>
    </tr>
    <tr>
      <th>148</th>
      <td>NaN</td>
      <td>Other</td>
      <td>LOTS 1 THRU 6, BLK 15, D.T. DENNY'S NO. SEATTL...</td>
      <td>Blanket Permit for interior non-structural alt...</td>
      <td>BLANKET</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.742238</td>
    </tr>
    <tr>
      <th>149</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>RELOCATE 125A SERVICE &amp; REMODEL TOP FLOOR. 7/2...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.773648</td>
    </tr>
    <tr>
      <th>150</th>
      <td>NaN</td>
      <td>none</td>
      <td>LT 6, BLK 7, PLEASANT RIDGE ADD TGW S 5' OF VA...</td>
      <td>DEMOLITION OF DUPLEX. ESTABLISH USE AS TWO 2-U...</td>
      <td>PTS PERMIT</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.853380</td>
    </tr>
    <tr>
      <th>151</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, GENERAL CONTRAC...</td>
      <td>reed_wright_htg_&amp;_elec_co_inc</td>
      <td>None</td>
      <td>INSTALL NEW STORAGE TANK</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.858485</td>
    </tr>
    <tr>
      <th>152</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>thyssenkrupp_elevator_corp</td>
      <td>None</td>
      <td>INSTALL NEW HYDRAULIC ELEVATOR - EU0334</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.977842</td>
    </tr>
    <tr>
      <th>153</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRING OF COMMERCIAL T.I 2176 SQ FT</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.224488</td>
    </tr>
    <tr>
      <th>154</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>mr_rooter_plumbing_(sposari's)</td>
      <td>None</td>
      <td>SIDE SEWER REPAIR</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.413594</td>
    </tr>
    <tr>
      <th>155</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>kemly_electric_incorporated</td>
      <td>None</td>
      <td>PROVIDE AND INSTALL 125 AMP SERVICE, PICKING U...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.803248</td>
    </tr>
    <tr>
      <th>156</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>TEMP OPERATING PERMIT</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.672957</td>
    </tr>
    <tr>
      <th>157</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL NEW DUCTWORK FROM EXISTING FAN COIL UN...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.118093</td>
    </tr>
    <tr>
      <th>158</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRE OFFICE, TI UNDER 2,500 SQ FT</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.220370</td>
    </tr>
    <tr>
      <th>159</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL OUTLETS &amp; LIGHTING FOR RESTAURANT SPACE</td>
      <td>ELECTRICAL</td>
      <td>COMMERCIAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.174320</td>
    </tr>
    <tr>
      <th>160</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INITIAL TENANT IMPROVEMENTS TO PORTION OF EXIS...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.566660</td>
    </tr>
    <tr>
      <th>161</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL NEW SPRINKLER MONITORING SYSTEM.</td>
      <td>ELECTRICAL</td>
      <td>INSTITUTIONAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.840863</td>
    </tr>
    <tr>
      <th>162</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>otis_elevator_company</td>
      <td>None</td>
      <td>60-DAY TEMPORARY OPERATING PERMIT. 10/2/03 JCB...</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.668335</td>
    </tr>
    <tr>
      <th>163</th>
      <td>GENERAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>REPLACE FURNACE &amp; DUCTWORK</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.718692</td>
    </tr>
    <tr>
      <th>164</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL SPRINKLERS IN EXISTING BLDG. PER PLANS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.108238</td>
    </tr>
    <tr>
      <th>165</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>AUTOMATIC CERTIFICATION</td>
      <td>BOILER / NEW OBJECT OR MONITOR'G CHANGE</td>
      <td>INSTITUTIONAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.413566</td>
    </tr>
    <tr>
      <th>166</th>
      <td>NaN</td>
      <td>Other</td>
      <td>None</td>
      <td>TELEPHONE CABLING FOR TRIBECA CONDOMINIUMS - T...</td>
      <td>ELECTRICAL</td>
      <td>MULTIFAMILY</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.093832</td>
    </tr>
    <tr>
      <th>167</th>
      <td>GENERAL CONTRACTOR LICENSE, REGIS. SIDE SEWER ...</td>
      <td>rescue_rooter_llc</td>
      <td>None</td>
      <td>REPAIR EXISTING SIDE SEWER.</td>
      <td>SIDE SEWER</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.691574</td>
    </tr>
    <tr>
      <th>168</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>ADDING (1) BEDROOM, (2) BATHROOMS, &amp; REMODEL E...</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.385920</td>
    </tr>
    <tr>
      <th>169</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT DETACHED GARAGE ACCESSORY TO EXISTIN...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.643829</td>
    </tr>
    <tr>
      <th>170</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>171</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>WIRE SINGLE FAMILY RESIDENCE.</td>
      <td>ELECTRICAL</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.979367</td>
    </tr>
    <tr>
      <th>172</th>
      <td>ELEVATOR CONTRACTOR LICENSE</td>
      <td>schindler_elevator_company</td>
      <td>None</td>
      <td>INSTALL NEW ESCALLATOR</td>
      <td>CONVEYANCE</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.468009</td>
    </tr>
    <tr>
      <th>173</th>
      <td>ELECTRICAL CONTRACTOR LICENSE, REFRIGERATION C...</td>
      <td>Other</td>
      <td>None</td>
      <td>INSTALL NEW GAS FURNACE AND DUCTWORK IN NEW CO...</td>
      <td>MECHANICAL /</td>
      <td>SINGLE FAMILY / DUPLEX</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.709800</td>
    </tr>
    <tr>
      <th>174</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CHANGE OF USE FROM RETAIL TO MEDICAL SERVICE O...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.488719</td>
    </tr>
    <tr>
      <th>175</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>RELOCATE 6 DIFFUSERS, REPLACE 20 RETURN AIR GR...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.690171</td>
    </tr>
    <tr>
      <th>176</th>
      <td>ELECTRICAL CONTRACTOR LICENSE</td>
      <td>Other</td>
      <td>None</td>
      <td>TELECOMMUNICATIONS UPGRADE - MCCLURE MIDDLE SC...</td>
      <td>ELECTRICAL</td>
      <td>INSTITUTIONAL</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.162305</td>
    </tr>
    <tr>
      <th>177</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>DUCTWORK AND DIFFUSERS FOR TENANT IMPROVEMENT ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.387021</td>
    </tr>
    <tr>
      <th>178</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>REMOVE ABANDONED DETACHED DRIVE-UP BANKING CAN...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.394036</td>
    </tr>
    <tr>
      <th>179</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL HVAC SYSTEM FOR CORRIDORS AND EXHAUST ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.475335</td>
    </tr>
    <tr>
      <th>180</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>FUTURE CONSTRUCION OF A MIXED-USE BLDG COMPRIS...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.787836</td>
    </tr>
    <tr>
      <th>181</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>DEMOLITION OF 3 COMMERCIAL BUILDINGS FUTURE CO...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.081658</td>
    </tr>
    <tr>
      <th>182</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INTERIOR ALTERATIONS TO BASEMENT SPACES FOR CO...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.888768</td>
    </tr>
    <tr>
      <th>183</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>TO ID 1(EXISTING 14 UNIT APT BLDG.) DEMOLISH 1...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.995050</td>
    </tr>
    <tr>
      <th>184</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT 2-STORY ADDITION &amp; DECK INTERIOR ALT...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.092820</td>
    </tr>
    <tr>
      <th>185</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>REPLACE POOL HEATING &amp; VENTILATING UNIT WITH N...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.954480</td>
    </tr>
    <tr>
      <th>186</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ALTERATIONS TO RESTROOM ON MAIN FLOOR &amp; CONSTR...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.778858</td>
    </tr>
    <tr>
      <th>187</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ON MUP 8801566 CONSTRUCT 5-STORY HOTEL WITH PA...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.140373</td>
    </tr>
    <tr>
      <th>188</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT RETAINING WALL ACC'Y TO SINGLE FAMIL...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.592336</td>
    </tr>
    <tr>
      <th>189</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INTERIOR ALTERATIONS TO FINISH BASEMENT OF SIN...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.001119</td>
    </tr>
    <tr>
      <th>190</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>INSTALL (1) SUPPLY DIFF.,(5) RETURN AIR, RELOC...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.538669</td>
    </tr>
    <tr>
      <th>191</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT DETACHED GARAGE ACCESSORY TO SINGLE ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.517993</td>
    </tr>
    <tr>
      <th>192</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>TO PORTION OF GARAGE FOR MOBILEPHONE EQUIPMENT...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.864883</td>
    </tr>
    <tr>
      <th>193</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ON MUP 8801566 CONSTRUCT 5-STORY HOTEL WITH PA...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.692185</td>
    </tr>
    <tr>
      <th>194</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>BLANKET PERMIT-INTR NON-STRUCTURAL ALTERATION ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.930322</td>
    </tr>
    <tr>
      <th>195</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>CONSTRUCT RETAINING WALL, PER PLANS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.860010</td>
    </tr>
    <tr>
      <th>196</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>REPLACE OLD PLASTER WITH NEW PLASTERBOARD AT I...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.031414</td>
    </tr>
    <tr>
      <th>197</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>BLANKET PERMIT-INTR NON-STRUCTURAL ALTERATION ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.930322</td>
    </tr>
    <tr>
      <th>198</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>REPLACE ROOF SHEATHING OF SFR, STFI.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.102156</td>
    </tr>
    <tr>
      <th>199</th>
      <td>NaN</td>
      <td>none</td>
      <td>None</td>
      <td>ENCLOSE REAR PORCH UNDER EXISTING ROOF ON SFR ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.667293</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.dtypes
```




    licensetype            object
    businessname           object
    legaldescription       object
    description            object
    type                   object
    subtype                object
    job_value             float64
    has_ld                  int64
    tf-idf_description    float64
    dtype: object
