#Load Libraries
library(tidyverse)
library(caret)
library(class)
library(dplyr)
library(glmnet)
library(pROC)
library(text2vec)
library(tm)
library(SnowballC)
library(randomForest)
library(gbm)
library(ISLR)
library(corrplot)
library(ggplot2)
library(reshape2)
library(stringr)
library(vip)

setwd("C:/Users/malun/Documents/UMD Grad Spring '24/BUDT 758T Data Mining and Predictive Analytics/BUDT 758T Final Project")
train_x <- read_csv("airbnb_train_x_2024.csv")
train_y <- read_csv("airbnb_train_y_2024.csv")
test_x <- read_csv("airbnb_test_x_2024.csv")

train_x$dataset <- "train"
test_x$dataset <- "test"
combined_data <- bind_rows(train_x, test_x)

################################# Clean Train/Test Data #################################
combined_data_cleaned <- combined_data %>%
  # Clean categorical features
  mutate(
    bed_type = factor(bed_type),
    cancellation_policy = factor(ifelse(cancellation_policy %in% c(
      "strict", "super_strict_30", "super_strict_60", "no_refunds"), "strict", cancellation_policy)),
    license = factor(ifelse(is.na(license), 0, 1))
  ) %>%
  mutate(market = factor(ifelse(is.na(market), "OTHER", market))) %>%
  mutate(
    property_type = factor(ifelse(is.na(property_type), "Apartment", property_type)),
    property_type = factor(case_when(
      property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
      property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
      property_type %in% c("Townhouse", "Condominium") ~ "condo",
      property_type %in% c("Bungalow", "House") ~ "house",
      TRUE ~ "other"
    )),
    room_type = factor(room_type),
    state = factor(toupper(state))
  ) %>%
  select(-city, -country, -country_code, -experiences_offered, -host_location,
         -host_neighbourhood, -host_response_time, -jurisdiction_names, -neighborhood,
         -neighborhood_group, -smart_location,
         -first_review) %>%
  # Clean numeric data
  mutate(
    accommodates = ifelse(is.na(accommodates), median(accommodates, na.rm = TRUE), accommodates),
    availability_365 = factor(ifelse(availability_365 > 0, 1, 0)),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    bedrooms = ifelse(is.na(bedrooms), median(bedrooms, na.rm = TRUE), bedrooms),
    beds = ifelse(is.na(beds), median(beds, na.rm = TRUE), beds),
    cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
    cleaning_fee = factor(ifelse(cleaning_fee > 0, 1,0)),
    extra_people = round(log(ifelse(is.na(extra_people), 0, extra_people)) ,0),
    extra_people = ifelse(extra_people <=0, 1, extra_people),
    host_acceptance = factor(case_when(
      host_acceptance_rate == 100 ~ "ALL",
      host_acceptance_rate < 100 ~ "HIGH",
      host_acceptance_rate < 50 ~ "MODERATE",
      TRUE ~ "MISSING")),
    # host_listings_count non-monotonic relationship with target label
    log_host_listings_count = round(log(ifelse(is.na(host_listings_count), 0, host_listings_count)), 0),
    HLC_bin = cut(log_host_listings_count, breaks = 6,
                  labels = c("HLC1", "HLC2", "HLC3", "HLC4", "HLC5", "HLC6")),
    host_response = factor(case_when(
      host_response_rate == 100 ~ "ALL",
      host_response_rate < 100 ~ "SOME",
      TRUE ~ "MISSING")),
    log_host_total_listings_count =  round(log(ifelse(is.na(host_total_listings_count), 0, host_total_listings_count)), 0),
    log_host_total_listings_count = ifelse(log_host_total_listings_count <= 0, 1, log_host_total_listings_count),
    log_maximum_nights = round(log(ifelse(is.na(maximum_nights), 0 ,maximum_nights)),0),
    log_maximum_nights = case_when(
      log_maximum_nights == 0 ~ 1,
      log_maximum_nights >= 7 ~ 6,
      TRUE ~ log_maximum_nights),
    log_minimum_nights = round(log(ifelse(is.na(minimum_nights), 0 ,minimum_nights)),0),
    log_minimum_nights = factor(case_when(
      log_minimum_nights >= 3 ~ 3,
      log_minimum_nights >= 1 ~ log_minimum_nights,
      log_minimum_nights <= 1 ~ 0,
      TRUE ~ log_minimum_nights)),
    monthly_price = ifelse(is.na(monthly_price), 0, monthly_price),
    monthly_price = ifelse(monthly_price < price*30 & monthly_price != 0, 1, 0),
    monthly_price = factor(ifelse(is.na(monthly_price), 0, monthly_price)),
    log_price = round(log(ifelse(is.na(price), median(price, na.rm = TRUE), price)),0),
    log_price = ifelse(log_price <= 0 , 1, log_price),
    security_deposit = ifelse(is.na(security_deposit), 0, security_deposit),
    security_deposit =  factor(ifelse(security_deposit > 0, 1, 0))
  ) %>%
  select(-availability_30, -availability_60, -availability_90,
         -square_feet, -weekly_price, -zipcode,
         -host_response_rate, -host_acceptance_rate, -host_listings_count,
         -host_total_listings_count, -maximum_nights, -minimum_nights, -price,
         -guests_included, -log_host_listings_count) %>%
  # wendy feature engineering
  mutate(
    host_since_days = as.integer(difftime(Sys.Date(), host_since, units = "days")),
    host_since_days = ifelse(is.na(host_since_days), median(host_since_days, na.rm = TRUE), host_since_days),
    description_length = ifelse(is.na(description), 0, nchar(as.character(description))),
    pets_allowed = factor(ifelse(grepl("Pets allowed", amenities), 1, 0))
  ) %>%
  # jerry feature engineering
  mutate(
    neighborhood_overview = ifelse(is.na(neighborhood_overview), "none", neighborhood_overview),
    neighborhood_overview = tolower(neighborhood_overview),
    is_market_enterain = factor(
      ifelse(grepl("whole foods|walmart|giants|lidl|aldi|groceries|grocery|
                   supermarket|bar|costco|target|music|restaurants|cafes|
                   coffee|food|shopping",neighborhood_overview),1, 0)),
    is_interaction = factor(ifelse(is.na(interaction), 0, 1)),
    house_rules = ifelse(is.na(interaction), "none", house_rules),
    house_rules = tolower(house_rules),
    house_rules_no_count = round(log(str_count(house_rules, "no")),0),
    house_rules_no_count = factor(case_when(
      house_rules_no_count >= 1 ~ 1,
      house_rules_no_count < 1 ~ 0,
      TRUE ~ 0
    ))) %>%
  # pravah feature engineering
  mutate(
    summary_length = ifelse(is.na(summary), 0, nchar(as.character(summary))),
    host_about = factor(ifelse(is.na(host_about), 0, 1)),
    house_rules = factor(ifelse(is.na(house_rules), 0, 1))
  ) %>%
  # gautam feature engineering
  mutate(amenities_count = sapply(strsplit(as.character(amenities), ","), length),
         host_verifications_count = sapply(strsplit(as.character(host_verifications), ","), length)
  ) %>%
  # clean text feature
  select(-access, -description, -features, -host_about, -host_name, -house_rules,
         -interaction, -name, -neighborhood_overview, -notes, -space, -street,
         -summary, -transit, -host_since, -amenities, -host_verifications)

# Define a function to determine timezone based on latitude and longitude
determine_timezone <- function(lat, long) {
  if (lat >= 24 & lat <= 47 & long >= -85 & long <= -67) {
    return("ET")
  } else if (lat >= 25 & lat <= 49 & long >= -103 & long <= -85) {
    return("CT")
  } else if (lat >= 29 & lat <= 49 & long >= -115 & long <= -102) {
    return("MT")
  } else if (lat >= 32 & lat <= 49 & long >= -125 & long <= -114) {
    return("PT")
  } else if (lat >= 54 & long >= -172 & long <= -130) {
    return("AKT")
  } else if (lat >= 19 & lat <= 28 & long >= -178 & long <= -154) {
    return("HAT")
  } else {
    return("OTHER")
  }
}

# Apply the function to create the timezone column
combined_data_cleaned$timezone <- mapply(determine_timezone, combined_data_cleaned$latitude, combined_data_cleaned$longitude)
combined_data_cleaned <- combined_data_cleaned %>%
  mutate(timezone = factor(timezone)) %>%
  select(-latitude, -longitude)

summary(combined_data_cleaned)
colnames(combined_data_cleaned)

################################ Normalize Combined Data ################################
min_accommodates <- min(combined_data_cleaned$accommodates)
max_accommodates <- max(combined_data_cleaned$accommodates)
combined_data_cleaned$accommodates <- (combined_data_cleaned$accommodates - min_accommodates) / (max_accommodates - min_accommodates)

min_bathrooms <- min(combined_data_cleaned$bathrooms)
max_bathrooms <- max(combined_data_cleaned$bathrooms)
combined_data_cleaned$bathrooms <- (combined_data_cleaned$bathrooms - min_bathrooms) / (max_bathrooms - min_bathrooms)

min_bedrooms <- min(combined_data_cleaned$bedrooms)
max_bedrooms <- max(combined_data_cleaned$bedrooms)
combined_data_cleaned$bedrooms <- (combined_data_cleaned$bathrooms - min_bedrooms) / (max_bedrooms - min_bedrooms)

min_beds <- min(combined_data_cleaned$beds)
max_beds <- max(combined_data_cleaned$beds)
combined_data_cleaned$beds <- (combined_data_cleaned$beds - min_beds) / (max_beds - min_beds)

min_extra_people <- min(combined_data_cleaned$extra_people)
max_extra_people <- max(combined_data_cleaned$extra_people)
combined_data_cleaned$extra_people <- (combined_data_cleaned$extra_people - min_extra_people) / (max_extra_people - min_extra_people)

min_log_host_total_listings_count <- min(combined_data_cleaned$log_host_total_listings_count)
max_log_host_total_listings_count <- max(combined_data_cleaned$log_host_total_listings_count)
combined_data_cleaned$log_host_total_listings_count <- (combined_data_cleaned$log_host_total_listings_count - min_log_host_total_listings_count) / (max_log_host_total_listings_count - min_log_host_total_listings_count)

min_log_maximum_nights <- min(combined_data_cleaned$log_maximum_nights)
max_log_maximum_nights <- max(combined_data_cleaned$log_maximum_nights)
combined_data_cleaned$log_maximum_nights <- (combined_data_cleaned$log_maximum_nights - min_log_maximum_nights) / (max_log_maximum_nights - min_log_maximum_nights)

min_host_since_days <- min(combined_data_cleaned$host_since_days)
max_host_since_days <- max(combined_data_cleaned$host_since_days)
combined_data_cleaned$host_since_days <- (combined_data_cleaned$host_since_days - min_host_since_days) / (max_host_since_days - min_host_since_days)

min_description_length <- min(combined_data_cleaned$description_length)
max_description_length <- max(combined_data_cleaned$description_length)
combined_data_cleaned$description_length <- (combined_data_cleaned$description_length - min_description_length) / (max_description_length - min_description_length)

min_summary_length <- min(combined_data_cleaned$summary_length)
max_summary_length <- max(combined_data_cleaned$summary_length)
combined_data_cleaned$summary_length <- (combined_data_cleaned$summary_length - min_summary_length) / (max_summary_length - min_summary_length)

summary(combined_data_cleaned)

################################# Clean Text Feature #################################
combined_data$id <- 1:nrow(combined_data)

cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers() %>%
    removePunctuation() %>%
    removeWords(tm::stopwords(kind = "en")) %>%
    word_tokenizer()
}

process_text_feature <- function(data_frame, feature_name, vocab_term_max = 10, term_count_min = NULL, doc_proportion_max = NULL) {
  # Replace separators with space
  data_frame[[feature_name]] <- gsub(",", " ", data_frame[[feature_name]])
  
  # Tokenize text
  it_train <- itoken(data_frame[[feature_name]],
                     preprocessor = tolower,
                     tokenizer = cleaning_tokenizer,
                     ids = data_frame$id,
                     progressbar = FALSE)
  
  # Create vocabulary
  vocab_train <- create_vocabulary(it_train)
  if (!is.null(term_count_min) || !is.null(doc_proportion_max)) {
    vocab_train <- prune_vocabulary(vocab_train, term_count_min = term_count_min, doc_proportion_max = doc_proportion_max)
  }
  vocab_small_train <- prune_vocabulary(vocab_train, vocab_term_max = vocab_term_max)
  vectorizer_train <- vocab_vectorizer(vocab_small_train)
  
  # Convert to DTM
  dtm_train <- create_dtm(it_train, vectorizer_train)
  dtm_train_df <- data.frame(as.matrix(dtm_train))
  
  return(list(train_dtm = dtm_train_df))
}

# Process amenities
amenities_result <- process_text_feature(combined_data, "amenities", vocab_term_max = 75)

# Process features
features_result <- process_text_feature(combined_data, "features", vocab_term_max = 75)

# Process description
description_result <- process_text_feature(combined_data, "description", vocab_term_max = 75)

# Process space
space_result <- process_text_feature(combined_data, "space", vocab_term_max = 75)

# Process summary
summary_result <- process_text_feature(combined_data, "summary", vocab_term_max = 75)

# Process name
name_result <- process_text_feature(combined_data, "name", vocab_term_max = 75)

# Process neighborhood
neighborhood_result<- process_text_feature(combined_data, "neighborhood_overview", vocab_term_max = 10)

# Process access
access_result<-process_text_feature(combined_data, "access", vocab_term_max = 10)

# Process host_about
host_about_result<- process_text_feature(combined_data, "host_about", vocab_term_max = 10)

# Process interaction
interaction_result<- process_text_feature(combined_data, "interaction", vocab_term_max = 10)

# Process notes
notes_result<-process_text_feature(combined_data, "notes", vocab_term_max = 10)

# Process transit
transit_result<-process_text_feature(combined_data, "transit", vocab_term_max = 10)

# Process house_rules
house_rules_result<-process_text_feature(combined_data, "house_rules", vocab_term_max = 10)

# Combine the results
combined_cleaned_text_feature <- cbind(amenities_result$train_dtm, features_result$train_dtm,
                                       description_result$train_dtm)

############################## Bind Text Feature & Remove Duplicates ##############################

combined_data_cleaned <- cbind(combined_data_cleaned, combined_cleaned_text_feature)


# Identify and remove duplicate column names
dup_cols <- which(duplicated(names(combined_data_cleaned)))
combined_data_cleaned <- combined_data_cleaned[, -dup_cols]

# State Mapping
state_mapping <- c("Alabama" = "AL",
                   "Alaska" = "AK",
                   "Arizona" = "AZ",
                   "Arkansas" = "AR",
                   "California" = "CA",
                   "Colorado" = "CO",
                   "Connecticut" = "CT",
                   "Delaware" = "DE",
                   "District of Columbia" = "DC",
                   "Florida" = "FL",
                   "Georgia" = "GA",
                   "Hawaii" = "HI",
                   "Idaho" = "ID",
                   "Illinois" = "IL",
                   "Indiana" = "IN",
                   "Iowa" = "IA",
                   "Kansas" = "KS",
                   "Kentucky" = "KY",
                   "Louisiana" = "LA",
                   "Maine" = "ME",
                   "Maryland" = "MD",
                   "Massachusetts" = "MA",
                   "Michigan" = "MI",
                   "Minnesota" = "MN",
                   "Mississippi" = "MS",
                   "Missouri" = "MO",
                   "Montana" = "MT",
                   "Nebraska" = "NE",
                   "Nevada" = "NV",
                   "New Hampshire" = "NH",
                   "New Jersey" = "NJ",
                   "New Mexico" = "NM",
                   "New York" = "NY",
                   "North Carolina" = "NC",
                   "North Dakota" = "ND",
                   "Ohio" = "OH",
                   "Oklahoma" = "OK",
                   "Oregon" = "OR",
                   "Pennsylvania" = "PA",
                   "Rhode Island" = "RI",
                   "South Carolina" = "SC",
                   "South Dakota" = "SD",
                   "Tennessee" = "TN",
                   "Texas" = "TX",
                   "Utah" = "UT",
                   "Vermont" = "VT",
                   "Virginia" = "VA",
                   "Washington" = "WA",
                   "West Virginia" = "WV",
                   "Wisconsin" = "WI",
                   "Wyoming" = "WY")

## external data 

############################## Happiness Data External ##############################
# https://worldpopulationreview.com/state-rankings/happiest-states
happiness<- read_csv("happiest-states-2024.csv")
happiness$state <- state_mapping[happiness$state]

# Identify the top 10 happiest states based on the lowest (best) ranks
top_10_states_happ <- happiness %>%
  arrange(HappiestStatesCommunityAndEnvironmentRank) %>%  # Ensure this sorts ascending
  slice(1:10) %>%
  pull(state)  # Use pull to directly extract the state column

# Add the new variable to combined_data_cleaned, using as.factor
combined_data_cleaned <- combined_data_cleaned %>%
  mutate(top10happy = ifelse(state %in% top_10_states_happ, 1, 0),  # Assign 1 for top 10, 0 otherwise
         top10happy = as.factor(top10happy))  # Convert to factor

## ############################## Population Data External ##############################
# https://worldpopulationreview.com/states

population <- read.csv("population info.csv")
population$state <- state_mapping[population$state]

# Identify the top 10 populous states
top_10_states_pop <- population %>%
  arrange(desc(pop_2024)) %>%
  slice(1:10) %>%
  pull(state)

# Add the new variable to combined_data_cleaned, using as.factor
combined_data_cleaned <- combined_data_cleaned %>%
  mutate(top10pop = ifelse(state %in% top_10_states_pop, 1, 0),
         top10pop = as.factor(top10pop))  # Convert to factor


############################## Weather Data External ##############################
# https://worldpopulationreview.com/state-rankings/best-weather-by-state

weather <- read.csv("best-weather-by-state-2024.csv")
weather$state <- state_mapping[weather$state]

# Identify the top 10 states based on the lowest extreme weather events
top_10_states_weather <- weather %>%
  arrange(StatesWithBestWeatherNumofExremeWeatherEvents) %>%
  slice(1:10) %>%
  pull(state)

# Add the new variable to combined_data_cleaned, using as.factor
combined_data_cleaned <- combined_data_cleaned %>%
  mutate(top10weather = ifelse(state %in% top_10_states_weather, 1, 0),
         top10weather = as.factor(top10weather))  # Convert to factor


############################## Crime Data External ##############################
# https://worldpopulationreview.com/state-rankings/best-weather-by-state

crime <- read.csv("crime-rate-by-state-2024.csv")
crime$state <- state_mapping[crime$state]

# View the structure of the data to understand the columns
str(crime)

# Assuming the crime rate column is named 'CrimeRate', adjust if necessary
# Calculate quantiles to define the categories
quantiles <- quantile(crime$CrimeRate, probs = c(0, 0.2, 0.4, 0.6, 0.8, 1))

# Define the crime category based on quantiles
crime$CrimeCategory <- cut(crime$CrimeRate,
                           breaks = c(-Inf, quantiles[2], quantiles[3], quantiles[4], quantiles[5], Inf),
                           labels = c("Very Low Crime", "Low Crime", "Moderate Crime", "High Crime", "Very High Crime"))

combined_data_cleaned <- combined_data_cleaned %>%
  mutate(crime_rate = ifelse(combined_data_cleaned$state == crime$state, crime$CrimeRate, "MISSING"),
         crime_rate = as.factor((crime_rate)))

### Split cleaned_data back into train_x and test_x based on 'dataset' column ###
train_x_cleaned <- combined_data_cleaned %>%
  filter(dataset == "train") %>%
  select(-dataset)
#summary(train_x_cleaned)

test_x_cleaned <- combined_data_cleaned %>%
  filter(dataset == "test") %>%
  select(-dataset)
#summary(test_x_cleaned)
################ Join the training y to the training x file ################
# turn the target variables into factors
train_cleaned <- cbind(train_x_cleaned, train_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score)) %>%
  select(-high_booking_rate)
################ EDA ###############

#1. market
ggplot(train_cleaned, aes(x = market, fill = factor(perfect_rating_score))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Proportion of Perfect Ratings by Market", x = "Market", y = "Proportion", fill = "Perfect Rating Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#2. host_response
ggplot(train_cleaned, aes(x = host_response, fill = factor(perfect_rating_score))) +
  geom_bar(position = "fill") +
  labs(
    title = "Proportion of Perfect Ratings by Host Response",
    x = "Host Response",
    y = "Proportion",
    fill = "Perfect Rating Score"
  ) +
  theme_minimal()

#3. host_since_days
ggplot(train_cleaned, aes(x = as.factor(perfect_rating_score), y = host_since_days, fill = as.factor(perfect_rating_score))) +
  geom_boxplot(outlier.color = "red") +
  labs(title = "Distribution of Host Since Days by Perfect Rating Score",
       x = "Perfect Rating Score",
       y = "Host Since Days",
       fill = "Perfect Rating Score")
scale_fill_manual(values = c("0" = "#FF6666", "1" = "#66B3FF"),
                  labels = c("0 (No)", "1 (Yes)")) +
  theme_minimal()


#4. summary_length
ggplot(train_cleaned, aes(x = perfect_rating_score, y = summary_length, fill = perfect_rating_score)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Box Plot of Summary Length by Perfect Rating Score",
       x = "Perfect Rating Score",
       y = "Summary Length (Normalized)",
       fill = "Perfect Rating Score") +
  scale_fill_manual(values = c("NO" = "#FF6666", "YES" = "#66B3FF"),
                    labels = c("NO", "YES")) +
  theme_minimal()

#5. monthly_price
ggplot(train_cleaned, aes(x = monthly_price, fill = perfect_rating_score)) +
  geom_bar(position = "dodge") +
  labs(title = "Monthly Price vs Perfect Rating Score",
       x = "Monthly Price (0 = Not Available, 1 = Available)",
       y = "Count",
       fill = "Perfect Rating Score") +
  scale_fill_manual(values = c("NO" = "#FF6666", "YES" = "#66B3FF"),
                    labels = c("NO", "YES")) +
  theme_minimal()

#6. availability_365
ggplot(train_cleaned, aes(x = availability_365, fill = perfect_rating_score)) +
  geom_bar(position = "dodge") +
  labs(title = "Availability 365 vs Perfect Rating Score",
       x = "Availability 365 (0 = Not Available, 1 = Available)",
       y = "Count",
       fill = "Perfect Rating Score") +
  scale_fill_manual(values = c("NO" = "#FF6666", "YES" = "#66B3FF"),
                    labels = c("NO", "YES")) +
  theme_minimal()

#7. description_length
ggplot(train_cleaned, aes(x = description_length, fill = perfect_rating_score)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density of Description Length by Perfect Rating Score",
       x = "Normalized Description Length",
       y = "Density",
       fill = "Perfect Rating Score") +
  theme_minimal()

#8. cancellation_policy
ggplot(train_cleaned, aes(x = cancellation_policy, fill = perfect_rating_score)) +
  geom_bar(position = "fill") +
  labs(
    title = "Proportion of Perfect Rating Scores by Cancellation Policy",
    x = "Cancellation Policy",
    y = "Proportion",
    fill = "Perfect Rating Score"
  ) +
  theme_minimal()

#9. log_price
ggplot(train_cleaned, aes(x = log_price, fill = perfect_rating_score)) +
  geom_density(alpha = 0.5) +
  labs(
    title = "Density Plot of Log Price by Perfect Rating Score",
    x = "Log Price",
    y = "Density",
    fill = "Perfect Rating Score"
  ) +
  theme_minimal()

#10. log_minimum_nights
ggplot(train_cleaned, aes(x = log_minimum_nights, fill = perfect_rating_score)) +
  geom_bar(position = "dodge") +
  labs(title = "Effect of Log Minimum Nights on Perfect Rating Score",
       x = "Log Minimum Nights",
       y = "Count",
       fill = "Perfect Rating Score") +
  theme_minimal()

################ cross validation for model selection ################
accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

set.seed(123)
# sample 10% train data
market_level <- unique(train_x$market)
cv_train_cleaned <- train_cleaned %>%
  sample_frac(0.1) %>%
  mutate(perfect_rating_score = factor(ifelse(perfect_rating_score == "YES", 1, 0)),
         market = ifelse(market %in% market_level, market, "OTHER"))

# shuffle data order
k <- 5
folds <- cut(seq(1,nrow(cv_train_cleaned)),breaks=k,labels=FALSE)

logistic_tpr_folds <- rep(0, k)
xbgoost_tpr_folds <- rep(0, k)
lasso_tpr_folds <- rep(0, k)
ridge_tpr_folds <- rep(0, k)
knn_trp_folds<-rep(0,k)
randomforest_tpr_folds<-rep(0,k)

for (i in 1:k) {
  # Define the validation fold indices
  valid_inds <- which(folds == i, arr.ind = TRUE)
  # Extract training fold
  train_fold <- cv_train_cleaned[-valid_inds, ]
  train_x_fold <- train_fold %>%
    mutate(market = factor(market),
           state = factor(state)) %>%
    select(-perfect_rating_score)
  train_y_fold <- train_fold %>%
    select(perfect_rating_score)
  market_level_cv <- unique(train_fold$market)
  
  # Extract validation fold
  valid_fold <- cv_train_cleaned[valid_inds, ]
  valid_fold <- valid_fold %>%
    mutate(market = factor(ifelse(market %in% market_level_cv, market, "OTHER")),
           state = factor(state))
  valid_x_fold <- valid_fold %>%
    select(-perfect_rating_score)
  valid_y_fold <- valid_fold %>%
    select(perfect_rating_score)
  
  ### logistic model on training fold
  logistic_model_cv <- glm(train_y_fold$perfect_rating_score ~ ., family = 'binomial', data = train_x_fold)
  # Predict on validation fold
  logistic_pred_on_valid <- predict(logistic_model_cv, newdata = valid_x_fold, type = "response")
  # Define cutoff for classification
  cutoff <- 0.5
  # Classify based on cutoff
  classification <- factor(ifelse(logistic_pred_on_valid > cutoff, 1, 0), levels = c("0", "1"))
  
  # Calculate confusion matrix based on current cutoff
  CM_tr <- confusionMatrix(data = classification,
                           reference = valid_y_fold$perfect_rating_score,
                           positive = "1")
  # Extract TP and FN from confusion matrix
  TP <- CM_tr$table[2, 2]
  FN <- CM_tr$table[1, 2]
  
  # Calculate True Positive Rate (TPR)
  current_TPR <- TP / (TP + FN)
  
  # Append TPR to logistic_tpr_folds
  logistic_tpr_folds[i] <- current_TPR
  
  ### xgboost
  train_y_fold <- train_y_fold %>%
    mutate(perfect_rating_score = as.numeric(perfect_rating_score),
           perfect_rating_score = ifelse(perfect_rating_score == 1, 0, 1))
  boost_model <- gbm(train_y_fold$perfect_rating_score~.,data = train_x_fold,
                     distribution = "bernoulli",
                     n.trees=100,
                     interaction.depth = 5)
  boost_pred <- predict(boost_model,
                        newdata = valid_x_fold,
                        type = 'response',
                        n.trees = 100)
  boost_class <- factor(ifelse(boost_pred > cutoff, 1, 0))
  # Calculate confusion matrix
  CM_tr_cv <- confusionMatrix(data = boost_class,
                              reference = valid_y_fold$perfect_rating_score,
                              positive = "1")
  TP <- CM_tr_cv$table[2, 2]
  FN <- CM_tr_cv$table[1, 2]
  current_TPR <- TP / (TP + FN)
  xbgoost_tpr_folds[i] <- current_TPR
  
  ### lasso
  lasso_model <- glmnet(as.matrix(train_x_fold), as.matrix(train_y_fold), family = "binomial", alpha = 1, lambda = 0.0004822898)
  lasso_pred_probabilities <- predict(lasso_model, newx = as.matrix(valid_x_fold), s = "lambda.min", type = "response")
  lasso_class <- factor(ifelse(lasso_pred_probabilities > cutoff, 1, 0), levels = c("0", "1"))
  CM_tr_lasso <- confusionMatrix(data = lasso_class,
                                 reference = valid_y_fold$perfect_rating_score,
                                 positive = "1")
  TP <- CM_tr_lasso$table[2, 2]
  FN <- CM_tr_lasso$table[1, 2]
  current_TPR <- TP / (TP + FN)
  lasso_tpr_folds[i] <- current_TPR
  
  ### ridge
  ridge_model <- glmnet(as.matrix(train_x_fold), as.matrix(train_y_fold), family = "binomial", alpha = 0, lambda = 0.001)
  ridge_pred_probabilities <- predict(ridge_model, newx = as.matrix(valid_x_fold), s = "lambda.min", type = "response")
  ridge_class <- factor(ifelse(ridge_pred_probabilities > cutoff, 1, 0), levels = c("0", "1"))
  CM_tr_ridge <- confusionMatrix(data = ridge_class,
                                 reference = valid_y_fold$perfect_rating_score,
                                 positive = "1")
  TP <- CM_tr_ridge$table[2, 2]
  FN <- CM_tr_ridge$table[1, 2]
  current_TPR <- TP / (TP + FN)
  ridge_tpr_folds[i] <- current_TPR
  
  ### Random Forest
  
  rf_model_cv <- randomForest(train_y_fold$perfect_rating_score ~ ., data = train_x_fold, ntree = 500, mtry = sqrt(ncol(train_x_fold)))
  rf_pred_cv <- predict(rf_model_cv, newdata = valid_x_fold, type = "response")
  rf_class <- factor(ifelse(rf_pred_cv > cutoff, 1, 0))
  CM_tr_rf <- confusionMatrix(data = rf_class,
                              reference = valid_y_fold$perfect_rating_score,
                              positive = "1")
  
  TP <- CM_tr_rf$table[2, 2]
  FN <- CM_tr_rf$table[1, 2]
  current_TPR <- TP / (TP + FN)
  randomforest_tpr_folds[i] <- current_TPR
  
  ### kNN
  
  knn_model_cv <- knn(train_x_fold, valid_x_fold, train_y_fold$perfect_rating_score, k = best_k)
  knn_pred_cv <- as.factor(knn_model_cv)
  CM_tr_knn <- confusionMatrix(data = knn_pred_cv,
                               reference = valid_y_fold$perfect_rating_score,
                               positive = "1")
  
  TP <- CM_tr_knn$table[2, 2]
  FN <- CM_tr_knn$table[1, 2]
  current_TPR <- TP / (TP + FN)
  knn_tpr_folds[i] <- current_TPR
  
}

cat("Average TPR on 5 fold-CV:", mean(logistic_tpr_folds), "\n")
cat("Average TPR on 5 fold-CV:", mean(xbgoost_tpr_folds), "\n")
cat("Average TPR on 5 fold-CV:", mean(lasso_tpr_folds), "\n")
cat("Average TPR on 5 fold-CV:", mean(ridge_tpr_folds), "\n")
cat("Average TPR on 5 fold-CV:", mean(randomforest_tpr_folds), "\n")
cat("Average TPR on 5 fold-CV:", mean(knn_tpr_folds), "\n")

# Combine the TPR vectors and create a data frame
k_fold <- c(1,2,3,4,5)
tpr_data <- data.frame(k_fold, logistic_tpr_folds, xbgoost_tpr_folds, lasso_tpr_folds, ridge_tpr_folds,knn_tpr_folds, randomforest_tpr_folds)
# Set the outer margin to make room for the legend
# Plotting
barplot(
  height = t(tpr_data[, -1]),  # Extracting TPR columns and transposing
  beside = TRUE,
  ylim = c(0, 0.5),
  main = "Comparison of TPR for Different Models across k_fold",
  names.arg = tpr_data$k_fold,
  xlab = "k_fold",
  ylab = "True Positive Rate",
  col = c("blue", "red", "orange", "grey","green","yellow"),
  legend.text = c("Logistic", "XGBoost", "Lasso", "Ridge", "KNN","Random Forest")
)
abline(h = mean(logistic_tpr_folds), col = "blue", lty = "dashed")
abline(h = mean(xbgoost_tpr_folds), col = "red", lty = "dashed")
abline(h = mean(lasso_tpr_folds), col = "orange", lty = "dashed")
abline(h = mean(ridge_tpr_folds), col = "grey", lty = "dashed")
abline(h = mean(knn_tpr_folds), col = "green", lty = "dashed")
abline(h = mean(randomforest_tpr_folds), col = "yellow", lty = "dashed")

###### Logistic Model#####

train_inst <- sample(nrow(train_cleaned), 0.70*nrow(train_cleaned))
train_x <- train_cleaned[train_inst,]
train_y <- train_x %>%
  select(perfect_rating_score) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score))
train_x <- train_x %>%
  select(-perfect_rating_score) %>%
  mutate(state = factor(state))
market_level <- unique(train_x$market)

valid_x <- train_cleaned[-train_inst,]
valid_y <- valid_x %>%
  select(perfect_rating_score) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score))
valid_x <- valid_x %>%
  select(-perfect_rating_score) %>%
  mutate(market = case_when(
    market %in% market_level ~ market,
    TRUE ~ "OTHER",
    
  ),
  state = factor(state))

cutoffs <- seq(0.01, 1.0, by = 0.01)
val_accs <- rep(0, length(cutoffs))
optimal_cutoff_lr <- NULL
max_TPR_lr <- 0
corresponding_FPR <- NULL
logistic_tpr <- rep(0, length(cutoffs))
logistic_fpr <- rep(0, length(cutoffs))

# Iterate through different cutoff values
logistic_model <- glm(train_y$perfect_rating_score~., family = 'binomial', data = train_x)
logistic_pred_on_valid <- predict(logistic_model, newdata = valid_x, type = "response")

for (i in c(1:length(cutoffs))) {
  cutoff = cutoffs[i]
  # Convert predicted probabilities to classification based on cutoff
  classification <- factor(ifelse(logistic_pred_on_valid > cutoff, "YES", "NO"),
                           levels = levels(train_y$perfect_rating_score))
  val_accs[i] <- accuracy(classification, valid_y$perfect_rating_score)
  
  # Calculate confusion matrix based on current cutoff
  CM_tr <- confusionMatrix(data = classification,
                           reference = valid_y$perfect_rating_score,
                           positive = "YES")
  
  # Extract TP, FN, TN, FP from confusion matrix
  TP <- CM_tr$table["YES", "YES"]
  FN <- CM_tr$table["NO", "YES"]
  TN <- CM_tr$table["NO", "NO"]
  FP <- CM_tr$table["YES", "NO"]
  
  # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
  current_TPR <- TP / (TP + FN)
  current_FPR <- FP / (TN + FP)
  logistic_tpr[i] <- current_TPR
  logistic_fpr[i] <- current_FPR
  
  # Check if current FPR is less than 10% and update optimal cutoff if TPR is higher
  if (current_FPR <= 0.1 && current_TPR > max_TPR) {
    optimal_cutoff_lr <- cutoff
    max_TPR_lr <- current_TPR
    corresponding_FPR_lr <- current_FPR
  }
}

# Print optimal cutoff and corresponding TPR, FPR
cat("Optimal Cutoff:", optimal_cutoff_lr, "\n")
cat("Corresponding TPR:", max_TPR_lr, "\n")
cat("Corresponding FPR:", corresponding_FPR_lr, "\n")

# train TPR accuracy
logistic_pred_on_train <- predict(logistic_model, newdata = train_x, type = "response")
classification <- factor(ifelse(logistic_pred_on_train > optimal_cutoff, 1, 0), levels = levels(valid_y$perfect_rating_score))
CM_tr <- confusionMatrix(data = classification,
                         reference = train_y$perfect_rating_score,
                         positive = "1")
CM_tr$table
TP <- CM_tr$table["1", "1"]
FN <- CM_tr$table["0", "1"]
TN <- CM_tr$table["0", "0"]
FP <- CM_tr$table["1", "0"]

TR_TPR <- TP / (TP + FN)
TR_FPR <- FP / (TN + FP)
TR_TPR
TR_FPR

logistic_roc <- data.frame(logistic_tpr, logistic_fpr)

# use all train data to prediction on test data

logistic_model <- glm(perfect_rating_score~., family = 'binomial', data = train_x_cleaned)
logistic_pred_on_test <- predict(logistic_model, newdata = test_x_cleaned, type = "response")
classifications_perfect_logistic <- factor(ifelse(logistic_pred_on_test > optimal_cutoff, "YES", "NO"))
assertthat::assert_that(sum(is.na(classifications_perfect_logistic))==0)
table(classifications_perfect_logistic)

write.table(classifications_perfect_logistic, "perfect_rating_score_group11_logistic.csv", row.names = FALSE)

###### XG Boost Model#####

train_y <- train_y %>%
  mutate(perfect_rating_score = ifelse(perfect_rating_score == 'NO', 0, 1))
valid_y <- valid_y %>%
  mutate(perfect_rating_score = ifelse(perfect_rating_score == 'NO', 0, 1))

boost.mod <- gbm(train_y$perfect_rating_score~.,data=train_x,
                 distribution="bernoulli",
                 n.trees=150,
                 interaction.depth=10)
boost_preds <- predict(boost.mod,
                       newdata=valid_x,
                       type='response',
                       n.trees=150)
xgboost_max_TPR <- 0
xgboost_FPR <- 0
xgboost_best_cutoff <- 0
xgboost_tpr <- rep(0, length(cutoffs))
xgboost_fpr <- rep(0, length(cutoffs))


for (i in c(1:length(cutoffs))) {
  cutoff <- cutoffs[i]
  # Convert predicted classes to factor with same levels as reference
  boost_class <- ifelse(boost_preds > cutoff, 1, 0)
  valid_y <- valid_y %>%
    mutate(perfect_rating_score = factor(perfect_rating_score))
  boost_class_factor <- factor(boost_class, levels = levels(valid_y$perfect_rating_score))
  
  # Calculate confusion matrix
  CM_tr <- confusionMatrix(data = boost_class_factor,
                           reference = valid_y$perfect_rating_score,
                           positive = "1")
  
  # Extract TP, FN, TN, FP from confusion matrix
  TP <- CM_tr$table["1", "1"]
  FN <- CM_tr$table["0", "1"]
  TN <- CM_tr$table["0", "0"]
  FP <- CM_tr$table["1", "0"]
  
  # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
  current_TPR <- TP / (TP + FN)
  current_FPR <- FP / (TN + FP)
  xgboost_tpr[i] <- current_TPR
  xgboost_fpr[i] <- current_FPR
  
  # Check if current cutoff produces higher TPR with FPR < 10%
  if (current_TPR > xgboost_max_TPR & current_FPR < 0.092) {
    xgboost_max_TPR <- current_TPR
    xgboost_best_cutoff <- cutof
    xgboost_FPR <- current_FPR 
  }
}

cat("Optimal Cutoff:", xgboost_best_cutoff, "\n")
cat("Corresponding TPR:", xgboost_max_TPR, "\n")
cat("Corresponding FPR:", xgboost_FPR, "\n")

# train TPR accuracy
boost_preds_tr <- predict(boost.mod,
                          newdata=train_x,
                          type='response',
                          n.trees=150)
boost_class_tr <- factor(ifelse(boost_preds_tr > xgboost_best_cutoff, 1, 0))
train_y <- train_y %>%
  mutate(perfect_rating_score = factor(perfect_rating_score))
CM_tr <- confusionMatrix(data = boost_class_tr,
                         reference = train_y$perfect_rating_score,
                         positive = "1")
TP <- CM_tr$table["1", "1"]
FN <- CM_tr$table["0", "1"]
TN <- CM_tr$table["0", "0"]
FP <- CM_tr$table["1", "0"]
TR_TPR <- TP / (TP + FN)
TR_FPR <- FP / (TN + FP)
TR_TPR
TR_FPR

xgboost_roc <- data.frame(xgboost_tpr, xgboost_fpr)

# use all train data to prediction on test data
train_x_feature <- train_x_cleaned %>%
  select(-perfect_rating_score)
train_x_label <- train_x_cleaned %>%
  select(perfect_rating_score) %>%
  mutate(perfect_rating_score = ifelse(perfect_rating_score == "YES", 1, 0))


boost.mod <- gbm(train_x_label$perfect_rating_score~.,data=train_x_feature,
                 distribution="bernoulli",
                 n.trees=150,
                 interaction.depth=10)

boost_preds_on_test <- predict(boost.mod,
                               newdata=test_x_cleaned,
                               type='response',
                               n.trees=150)

classifications_perfect_boost <- ifelse(boost_preds_on_test>0.4665, "YES", "NO")
assertthat::assert_that(sum(is.na(classifications_perfect_boost))==0)
table(classifications_perfect_boost)

write.table(classifications_perfect_boost, "perfect_rating_score_group11_boost_final.csv", row.names = FALSE)

# TPR/ FPR curve
plot(xgboost_fpr, xgboost_tpr, type = "l",
     xlab = "False Positive Rate (FPR)",
     ylab = "True Positive Rate (TPR)",
     col = "lightblue",
     lwd = 2,
     main = "ROC Curve for XGBoost")
abline(a = 0, b = 1, col = "grey4", lty = 2)
grid(col = "grey", lty = "dotted")

###### KNN Model #####

# Data Splitting
train_inst <- sample(nrow(train_x_cleaned), 0.70 * nrow(train_x_cleaned))
train_x <- train_x_cleaned[train_inst, ]
train_y <- train_x$perfect_rating_score
train_x <- train_x[, -which(names(train_x) == "perfect_rating_score")]

valid_x <- train_x_cleaned[-train_inst, ]
valid_y <- valid_x$perfect_rating_score
valid_x <- valid_x[, -which(names(valid_x) == "perfect_rating_score")]

# Target Variable Transformation
train_y <- ifelse(train_y == "NO", 0, 1)
valid_y <- ifelse(valid_y == "NO", 0, 1)

# Define evaluate_knn function
evaluate_knn <- function(train_x, train_y, valid_x, valid_y, k) {
  knn_model <- knn(train_x, valid_x, train_y, k = k)
  confusion_matrix <- table(knn_model, valid_y)
  TN <- confusion_matrix[1, 1]
  FP <- confusion_matrix[1, 2]
  FN <- confusion_matrix[2, 1]
  TP <- confusion_matrix[2, 2]
  FPR <- FP / (FP + TN)  # False Positive Rate
  TPR <- TP / (TP + FN)  # True Positive Rate
  accuracy_knn <- sum(diag(confusion_matrix)) / sum(confusion_matrix)  # Accuracy
  return(list(FPR = FPR, TPR = TPR, accuracy = accuracy))
}

# KNN Model Training

best_k <- NULL
best_FPR_knn <- 1

for (k in 1:ncol(train_x)) {
  evaluation <- evaluate_knn(train_x, train_y, valid_x, valid_y, k)
  FPR <- evaluation$FPR
  TPR <- evaluation$TPR
  accuracy <- evaluation$accuracy_knn
  
  if (FPR < 0.095) {
    if (FPR < best_FPR) {
      best_k <- k
      best_FPR_knn <- FPR
      best_TPR_knn <- TPR
      best_accuracy_knn <- accuracy
    }
  }
}

cat("Best k:", best_k, "\n")
cat("Corresponding FPR:", best_FPR, "\n")
cat("Corresponding TPR:", best_TPR, "\n")
cat("Accuracy:", best_accuracy, "\n")

# Train KNN with the best k
knn_model <- knn(train_x, valid_x, train_y, k = best_k)

#Predict with the best knn model
knn_pred <- predict(knn_model, test_x_cleaned)

classifications_perfect_knn <- ifelse(knn_pred>0.5, "YES", "NO")
assertthat::assert_that(sum(is.na(classifications_perfect_knn))==0)
table(classifications_perfect_knn)

write.table(classifications_perfect_knn, "perfect_rating_score_group11_knn.csv", row.names = FALSE)

###### Random Forest #####

# Data Splitting
train_inst <- sample(nrow(train_x_cleaned), 0.70 * nrow(train_x_cleaned))
train_x <- train_x_cleaned[train_inst, ]
train_y <- ifelse(train_x$perfect_rating_score == "NO", 0, 1)
train_x <- train_x[, -names(train_x) %in% c("perfect_rating_score")]

valid_x <- train_x_cleaned[-train_inst, ]
valid_y <- ifelse(valid_x$perfect_rating_score == "NO", 0, 1)
valid_x <- valid_x[, -names(valid_x) %in% c("perfect_rating_score")]

# Train Random Forest
rf_model <- randomForest(train_y ~ ., data = train_x, ntree = 500, mtry = sqrt(ncol(train_x)))

# Make predictions on validation set
rf_pred <- predict(rf_model, newdata = valid_x, type = "response")

# Initialize variables for optimal cutoff and corresponding TPR and FPR
optimal_cutoff_rf <- NULL
max_TPR_rf <- 0
corresponding_FPR_rf <- NULL

# Loop to find optimal cutoff
for (cutoff in seq(0.01, 0.99, by = 0.01)) {
  # Make predictions on validation set
  rf_pred_valid <- predict(rf_model, newdata = valid_x, type = "prob")[, 2] > cutoff
  
  # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
  conf_matrix_rf <- table(rf_pred_valid, valid_y)
  TN <- conf_matrix[1, 1]
  FP <- conf_matrix[1, 2]
  FN <- conf_matrix[2, 1]
  TP <- conf_matrix[2, 2]
  FPR <- FP / (FP + TN)
  TPR <- TP / (TP + FN)
  
  # Check if current FPR is less than 9.5% and update optimal cutoff if TPR is higher
  if (FPR < 0.095 && TPR > max_TPR_rf) {
    max_TPR_rf <- TPR
    corresponding_FPR_rf <- FPR
    optimal_cutoff_rf <- cutoff
  }
}

# Output optimal cutoff and corresponding TPR and FPR
cat("Optimal Cutoff:", optimal_cutoff_rf, "\n")
cat("Corresponding TPR:", max_TPR_rf, "\n")
cat("Corresponding FPR:", corresponding_FPR_rf, "\n")

rf_pred <- predict(rf_model, newdata = test_x_cleaned, type = "response")

classifications_perfect_rf <- ifelse(rf_pred>optimal_cutoff_rf, "YES", "NO")
assertthat::assert_that(sum(is.na(classifications_perfect_rf))==0)
table(classifications_perfect_rf)

write.table(classifications_perfect_rf, "perfect_rating_score_group11_rf.csv", row.names = FALSE)


###### Lasso Model #####

# Set seed for reproducibility
set.seed(123)

# Sample indices for creating training and validation sets
train_indices <- sample(nrow(train_cleaned), 0.70 * nrow(train_cleaned))
train_data <- train_cleaned[train_indices, ]
valid_data <- train_cleaned[-train_indices, ]

# Ensure all 'market' levels present in validation are in training
market_level <- unique(train_data$market)
valid_data <- valid_data %>%
  mutate(market = case_when(
    market %in% market_level ~ market,
    TRUE ~ "OTHER"
  ))

# Create the full model matrix from the adjusted data, then split
full_data <- bind_rows(train_data, valid_data) # Bind rows to keep structure
full_matrix <- model.matrix(~ . - 1 - perfect_rating_score, data = full_data)

# Number of train data rows may change due to filtering, recalculate it
train_n <- nrow(train_data)

# Split the full matrix into training and validation matrices
x_train <- full_matrix[1:train_n, ]
x_valid <- full_matrix[(train_n + 1):nrow(full_matrix), ]

# Prepare the response variable
y_full <- as.numeric(as.factor(full_data$perfect_rating_score)) - 1

# Split the response variable into training and validation
y_train <- y_full[1:train_n]
y_valid <- y_full[(train_n + 1):nrow(full_matrix)]

# Fit the model using cross-validation to find the best lambda
cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
# Extract the lambda values and corresponding cross-validated errors
lambda_values <- cv_model$lambda
cv_errors <- cv_model$cvm

# Plotting model complexity vs. error
plot_data <- data.frame(Lambda = lambda_values, Error = cv_errors)

ggplot(plot_data, aes(x = log(Lambda), y = Error)) +
  geom_line() +
  geom_point(aes(color = Lambda), size = 2) +
  scale_color_continuous(trans = 'reverse') +
  labs(x = "Log(Lambda) - Model Complexity", y = "Cross-Validated Error",
       title = "Error vs. Model Complexity for Lasso Model") +
  theme_minimal()

# Extract the best lambda and fit the final model
best_lambda <- cv_model$lambda.min
lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Predict probabilities using the validation data
lasso_pred_probabilities <- predict(lasso_model, newx = x_valid, s = "lambda.min", type = "response")

# Initialize variables to store optimal cutoff and performance metrics
optimal_cutoff_lasso <- 0
max_TPR_lasso <- 0
corresponding_FPR_lasso <- 0

# Define range of cutoff values for the Lasso model
cutoff_range_lasso <- seq(0.001, 0.999, by = 0.001)

# Nested loop to iterate through all cutoff values
for (cutoff_lasso in cutoff_range_lasso) {
  # Make predictions for the Lasso model using the current cutoff value
  predictions_lasso <- ifelse(lasso_pred_probabilities > cutoff_lasso, "YES", "NO")
  
  # Calculate confusion matrix based on Lasso model predictions
  CM_lasso <- confusionMatrix(data = factor(predictions_lasso, levels = c("YES", "NO")),
                              reference = as.factor(valid_data$perfect_rating_score),  # Use valid_data for reference
                              positive = "YES")
  
  # Extract TP, FN, TN, FP from confusion matrix
  TP <- CM_lasso$table["YES", "YES"]
  FN <- CM_lasso$table["NO", "YES"]
  TN <- CM_lasso$table["NO", "NO"]
  FP <- CM_lasso$table["YES", "NO"]
  
  # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
  current_TPR_lasso <- TP / (TP + FN)
  current_FPR_lasso <- FP / (TN + FP)
  
  # Check if current FPR is less than 10% and update optimal cutoff if TPR is higher
  if (current_FPR < 0.095 && current_TPR > max_TPR) {
    max_TPR_lasso <- current_TPR
    corresponding_FPR_lasso <- current_FPR
    optimal_cutoff_lasso <- cutoff_lasso
  }
}

# Print optimal cutoff and performance metrics for train-valid split
print("Train-Valid Split Model:")
print(paste("Optimal Cutoff for Lasso Model:", optimal_cutoff_lasso))
print(paste("Max TPR with FPR < 10%:", max_TPR_lasso))
print(paste("Corresponding FPR:", corresponding_FPR_lasso))

classifications_perfect_lasso <- ifelse(predictions_lasso>optimal_cutoff_lasso, "YES", "NO")
assertthat::assert_that(sum(is.na(classifications_perfect_lasso))==0)
table(classifications_perfect_lasso)

write.table(classifications_perfect_lasso, "perfect_rating_score_group11_lasso.csv", row.names = FALSE)


###### Ridge Model #####

#Run train validation on full model 
# Set seed for reproducibility
set.seed(123)

# Sample indices for creating training and validation sets
train_indices <- sample(nrow(train_cleaned), 0.70 * nrow(train_cleaned))
train_data <- train_cleaned[train_indices, ]
valid_data <- train_cleaned[-train_indices, ]

# Ensure all 'market' levels present in validation are in training
market_level <- unique(train_data$market)
valid_data <- valid_data %>%
  mutate(market = case_when(
    market %in% market_level ~ market,
    TRUE ~ "OTHER"
  ))

# Create the full model matrix from the adjusted data, then split
full_data <- bind_rows(train_data, valid_data) # Bind rows to keep structure
full_matrix <- model.matrix(~ . - 1 - perfect_rating_score, data = full_data)

# Number of train data rows may change due to filtering, recalculate it
train_n <- nrow(train_data)

# Split the full matrix into training and validation matrices
x_train <- full_matrix[1:train_n, ]
x_valid <- full_matrix[(train_n + 1):nrow(full_matrix), ]

# Prepare the response variable
y_full <- as.numeric(as.factor(full_data$perfect_rating_score)) - 1

# Split the response variable into training and validation
y_train <- y_full[1:train_n]
y_valid <- y_full[(train_n + 1):nrow(full_matrix)]

# Fit the model using cross-validation to find the best lambda
cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
plot(cv_model)

# Extract the best lambda and fit the final model
best_lambda <- cv_model$lambda.min
ridge_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0, lambda = best_lambda)

# Predict probabilities using the validation data
ridge_pred_probabilities <- predict(ridge_model, newx = x_valid, s = "lambda.min", type = "response")

# Initialize variables to store optimal cutoff and performance metrics
optimal_cutoff_ridge <- 0
max_TPR_ridge <- 0
corresponding_FPR_ridge <- 0

# Define range of cutoff values for the ridge model
cutoff_range_ridge <- seq(0.001, 0.999, by = 0.001)

# Nested loop to iterate through all cutoff values
for (cutoff_ridge in cutoff_range_ridge) {
  # Make predictions for the ridge model using the current cutoff value
  predictions_ridge <- ifelse(ridge_pred_probabilities > cutoff_ridge, "YES", "NO")
  
  # Calculate confusion matrix based on ridge model predictions
  CM_ridge <- confusionMatrix(data = factor(predictions_ridge, levels = c("YES", "NO")),
                              reference = as.factor(valid_data$perfect_rating_score),  # Use valid_data for reference
                              positive = "YES")
  
  # Extract TP, FN, TN, FP from confusion matrix
  TP <- CM_ridge$table["YES", "YES"]
  FN <- CM_ridge$table["NO", "YES"]
  TN <- CM_ridge$table["NO", "NO"]
  FP <- CM_ridge$table["YES", "NO"]
  
  # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
  current_TPR <- TP / (TP + FN)
  current_FPR <- FP / (TN + FP)
  
  # Check if current FPR is less than 10% and update optimal cutoff if TPR is higher
  if (current_FPR < 0.095 && current_TPR > max_TPR) {
    max_TPR_ridge <- current_TPR
    corresponding_FPR_ridge <- current_FPR
    optimal_cutoff_ridge <- cutoff_ridge
  }
}

# Print optimal cutoff and performance metrics for train-valid split
print("Train-Valid Split Model:")
print(paste("Optimal Cutoff for ridge Model:", optimal_cutoff_ridge))
print(paste("Max TPR with FPR < 10%:", max_TPR_ridge))
print(paste("Corresponding FPR:", corresponding_FPR_ridge))

## Check if external data variables improve model
# Exclude 'top10happy' from the model matrix
full_matrix_without <- model.matrix(~ . - 1 - perfect_rating_score - top10happy - top10pop - crime_rate - top10weather, data = full_data)

# Split the response variable into training and validation as before
x_train_without <- full_matrix_without[1:train_n, ]
x_valid_without <- full_matrix_without[(train_n + 1):nrow(full_matrix_without), ]

# Fit the model without 'top10happy'
cv_model_without <- cv.glmnet(x_train_without, y_train, family = "binomial", alpha = 0)
best_lambda_without <- cv_model_without$lambda.min
ridge_model_without <- glmnet(x_train_without, y_train, family = "binomial", alpha = 0, lambda = best_lambda_without)

# Predict probabilities using the validation data for the model without 'top10happy'
ridge_pred_probabilities_without <- predict(ridge_model_without, newx = x_valid_without, s = "lambda.min", type = "response")

# Calculate performance metrics at the same optimal cutoff determined previously
predictions_ridge_without <- ifelse(ridge_pred_probabilities_without > optimal_cutoff, "YES", "NO")
CM_ridge_without <- confusionMatrix(data = factor(predictions_ridge_without, levels = c("YES", "NO")),
                                    reference = as.factor(valid_data$perfect_rating_score),
                                    positive = "YES")

# Extract TP, FN, TN, FP from confusion matrix for the model without 'top10happy'
TP_without <- CM_ridge_without$table["YES", "YES"]
FN_without <- CM_ridge_without$table["NO", "YES"]
TN_without <- CM_ridge_without$table["NO", "NO"]
FP_without <- CM_ridge_without$table["YES", "NO"]

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for the model without external
TPR_without <- TP_without / (TP_without + FN_without)
FPR_without <- FP_without / (TN_without + FP_without)

# Print results for comparison
print("Model Comparison at Optimal Cutoff:")
print(paste("Optimal Cutoff Used:", optimal_cutoff))
print("With external:")
print(paste("TPR:", max_TPR, "FPR:", corresponding_FPR))
print("Without external:")
print(paste("TPR:", TPR_without, "FPR:", FPR_without))

## Predict on full train data
# Ensure all 'market' levels present in validation are in training
market_level_full <- unique(train_cleaned$market)
test_x_cleaned<- test_x_cleaned %>%
  mutate(market = case_when(
    market %in% market_level_full ~ market,
    TRUE ~ "OTHER"
  ))

train_cleaned_test<-train_cleaned%>%
  select(-perfect_rating_score)

# Create model matrices
full_data_test <- bind_rows(train_cleaned_test, test_x_cleaned)
full_matrix_test <- model.matrix(~ . - 1, data = full_data_test) # Now not excluding 'perfect_rating_score' here because it's already removed

# Define indices for train and test data within the combined matrix
train_n_full <- nrow(train_cleaned_test)
x_train_test <- full_matrix_test[1:train_n_full, ]
x_test <- full_matrix_test[(train_n_full + 1):nrow(full_matrix_test), ]

# Prepare the response variable, ensuring it's correctly specified
y_train_test <- as.numeric(train_cleaned$perfect_rating_score) - 1 # Assuming 'perfect_rating_score' is still a factor

# Re-train the model on the full training data
cv_model_full <- cv.glmnet(x_train_test, y_train_test, family = "binomial", alpha = 0)
best_lambda_full <- cv_model_full$lambda.min
ridge_model_full <- glmnet(x_train_test, y_train_test, family = "binomial", alpha = 0, lambda = best_lambda_full)

# Predict on the test data
ridge_pred_on_test <- predict(ridge_model_full, newx = x_test, s = "lambda.min", type = "response")

# Convert probabilities to factor based on optimal cutoff
classifications_perfect_ridge <- factor(ifelse(ridge_pred_on_test > optimal_cutoff, "YES", "NO"))

# Check for missing values
assertthat::assert_that(sum(is.na(classifications_perfect_ridge)) == 0)

# View the distribution of predicted classes
table(classifications_perfect_ridge)

write.table(classifications_perfect_ridge, "perfect_rating_score_group11_ridge.csv", row.names = FALSE)

########### Ridge and Lasso Learning Curve at 0.5 Cut-off ###########
accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}
set.seed(123)

# Split the initial dataset into 70% train and 30% validation
initial_train_indices <- createDataPartition(y = train_cleaned$perfect_rating_score, p = 0.7, list = FALSE)
train_data_initial <- train_cleaned[initial_train_indices, ]
valid_data <- train_cleaned[-initial_train_indices, ]

# Prepare a sequence of training sizes from the 70% training data
train_sizes <- seq(0.1, 1, by = 0.1) * nrow(train_data_initial)
accuracy_lasso <- numeric(length(train_sizes))
accuracy_ridge <- numeric(length(train_sizes))

# Loop over the sequence to generate models and calculate TPR
for (i in seq_along(train_sizes)) {
  # Sample indices for creating training sets of different sizes from the 70% subset
  train_indices <- sample(nrow(train_data_initial), train_sizes[i])
  train_data <- train_data_initial[train_indices, ]
  
  # Ensure all 'market' levels present in validation are in training
  market_level <- unique(train_data$market)
  valid_data <- valid_data %>%
    mutate(market = case_when(
      market %in% market_level ~ market,
      TRUE ~ "OTHER"
    ))
  # Create the full model matrix from the adjusted data, then split
  full_data <- bind_rows(train_data, valid_data) # Bind rows to keep structure
  full_matrix <- model.matrix(~ . - 1 - perfect_rating_score, data = full_data)
  
  # Number of train data rows may change due to filtering, recalculate it
  train_n <- nrow(train_data)
  
  # Split the full matrix into training and validation matrices
  x_train <- full_matrix[1:train_n, ]
  x_valid <- full_matrix[(train_n + 1):nrow(full_matrix), ]
  
  # Prepare the response variable
  y_full <- as.numeric(as.factor(full_data$perfect_rating_score)) - 1
  
  # Split the response variable into training and validation
  y_train <- y_full[1:train_n]
  y_valid <- y_full[(train_n + 1):nrow(full_matrix)]
  
  # Lasso model and accuracy
  lasso_cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
  best_lambda_lasso <- lasso_cv_model$lambda.min
  lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda_lasso)
  lasso_pred_probabilities <- predict(lasso_model, newx = x_valid, s = "lambda.min", type = "response")
  lasso_predictions <- ifelse(lasso_pred_probabilities > 0.5, 1, 0)
  lasso_accuracy_list[i]<- accuracy(lasso_predictions, y_valid)
  # ridge model and accuracy
  ridge_cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
  best_lambda_ridge <- ridge_cv_model$lambda.min
  ridge_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0, lambda = best_lambda_ridge)
  ridge_pred_probabilities <- predict(ridge_model, newx = x_valid, s = "lambda.min", type = "response")
  ridge_predictions <- ifelse(ridge_pred_probabilities > 0.5, 1, 0)
  ridge_accuracy_list[i]<- accuracy(ridge_predictions, y_valid)
}

# Create a dataframe for plotting
plot_data <- data.frame(
  TrainSize = rep(train_sizes, 2),
  Accuracy = c(lasso_accuracy_list, ridge_accuracy_list),
  Model = rep(c("Lasso", "Ridge"), each = length(train_sizes))
)

# Plot the learning curve with Accuracy for both models
ggplot(plot_data, aes(x = TrainSize, y = Accuracy, color = Model)) +
  geom_line() +
  geom_point() +
  labs(x = "Training Size", y = "Accuracy",
       title = "Learning Curve of Lasso and Ridge Models (Accuracy)") +
  theme_minimal()+
  theme(panel.grid  = (element_line(color = "grey95")))

######## Model Complexity vs Error - Lasso vs Ridge ############

# Extract the lambda values and corresponding cross-validated errors
cv_model_full_lasso <- cv.glmnet(x_train_test, y_train_test, family = "binomial", alpha = 1)
lambda_values_lasso <- cv_model_full_lasso$lambda
cv_errors_lasso <- cv_model_full_lasso$cvm

cv_model_full_ridge <- cv.glmnet(x_train_test, y_train_test, family = "binomial", alpha = 0)
lambda_values_ridge <- cv_model_full_ridge$lambda
cv_errors_ridge <- cv_model_full_ridge$cvm

# Prepare the data frames
plot_data_lasso <- data.frame(Model = "Lasso", Lambda = lambda_values_lasso, Error = cv_errors_lasso)
plot_data_ridge <- data.frame(Model = "Ridge", Lambda = lambda_values_ridge, Error = cv_errors_ridge)

# Combine the data frames
plot_data <- rbind(plot_data_lasso, plot_data_ridge)

# Plotting model complexity vs. error
ggplot(plot_data, aes(x = log(Lambda), y = Error, color = Model)) +
  geom_line() +
  geom_point(size = 2) +
  scale_color_manual(values = c("Lasso" = "#FF6666", "Ridge" = "#66B3FF")) +
  labs(x = "Log(Lambda) - Model Complexity", y = "Cross-Validated Error",
       title = "Error vs. Model Complexity for Lasso and Ridge Models") +
  theme_minimal()

