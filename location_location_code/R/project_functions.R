sales_df_reformatting <- function(sales_df) {
  sales_df[is.na(sales_df)] <- 0
  sales_df <- sales_df[!duplicated(sales_df$store_code),]
  DATE <- as.Date(colnames(sales_df[,-1]), format = "X%m.%d.%y")
  sales_mx <- t(sales_df[,-1])
  new_sales_df <- as.data.frame(sales_mx, row.names = NULL)
  colnames(new_sales_df) <- sales_df$store_code
  new_sales_df$DATE <- DATE
  new_sales_df$DATE <-
    paste(lubridate::year(new_sales_df$DATE),
          lubridate::isoweek(new_sales_df$DATE))
  new_sales_df <-
    aggregate(. ~ DATE, data = new_sales_df, sum, drop = F)
  return(new_sales_df)
}

extract_store_amenities_count <- function(store_code,
                                          surroundings_json,
                                          amenities_ls,
                                          store_codes_json,
                                          reviews_weighted) {
  store_filter <- store_code == store_codes_json
  if (any(store_filter)) {
    in_scope_surroundings_json <- surroundings_json[store_filter][[1]]
    
    amenities_count_data <- sapply(amenities_ls,
                                   function(amenity) {
                                     amenity_data <-
                                       in_scope_surroundings_json$surroundings[amenity][[1]]
                                     if (reviews_weighted) {
                                       n_amenity <-
                                         sapply(amenity_data, function(x)
                                           length(x$reviews) + 1)
                                     } else {
                                       n_amenity <- length(amenity_data)
                                     }
                                     return(n_amenity)
                                   })
  } else {
    amenities_count_data <- NULL
  }
  
  return(amenities_count_data)
}

create_amenities_count_df <- function(store_codes,
                                      surroundings_json,
                                      amenities_ls,
                                      store_codes_json,
                                      reviews_weighted = F) {
  
  store_code_amenities_counts <- lapply(store_codes, function(x)
    
    extract_store_amenities_count(
      store_code = x,
      surroundings_json = surroundings_json,
      amenities_ls = amenities_ls,
      store_codes_json = store_codes_json,
      reviews_weighted = reviews_weighted
    ))
  
  store_code_amenities_mtx <-
    Reduce(function(x, y)
      rbind(x, y), store_code_amenities_counts)
  
  store_code_amenities_df <- as.data.frame(store_code_amenities_mtx)
  
  row.names(store_code_amenities_df) <-
    1:nrow(store_code_amenities_df)
  
  store_code_amenities_df$n_amenities <-
    apply(store_code_amenities_df,
          1, sum)
  
  store_code_amenities_df$STORE_CODE <- store_codes[store_codes %in% store_codes_json]
  
  store_code_amenities_df <- subset(store_code_amenities_df,
                                    !is.na(STORE_CODE))
  
  return(store_code_amenities_df)
  
}


create_target_variables_df <- function(sales_df) {
  estimated_first_day <-
    apply(sales_df[, -1], 2, function(x)
      min(which(x > 0))) + 4
  
  average_sales <-
    apply(tail(sales_df[nrow(sales_df), -1], 24), 2, mean)
  
  sales_sd <- mapply(function(x, y)
    sd(sales_df[y:length(x), x]),
    x = colnames(sales_df[, -1]),
    y = estimated_first_day)
  
  weeks_onsale <- nrow(sales_df) - estimated_first_day
  
  sales_data_df <- data.frame(
    STORE_CODE = colnames(sales_df[, -1]),
    AVG_SALES = average_sales,
    SALES_VOLATILITY = sales_sd,
    WEEKS_ONSALE = weeks_onsale
  )
  
  sales_data_df$NOISY_POS <-
    abs(sales_data_df$SALES_VOLATILITY / sales_data_df$AVG_SALES)
  sales_data_df$NOISY_POS <-
    ifelse(is.finite(sales_data_df$NOISY_POS),
           sales_data_df$NOISY_POS,
           100)
  outliers_filter <- !(abs(scale(sales_data_df$AVG_SALES)) > 2)
  
  sales_data_df <-
    subset(sales_data_df, WEEKS_ONSALE > 8 & outliers_filter &
             AVG_SALES > 0  & SALES_VOLATILITY > 0 & !is.na(SALES_VOLATILITY))
  
  return(sales_data_df)
}


create_model_diagnostics_obj <- function(formula,
                                         no_modelling_vars = c("STORE_CODE",
                                                               "SALES_VOLATILITY",
                                                               "WEEKS_ONSALE",
                                                               "NOISY_POS",
                                                               "n_amenities"),
                                         method,
                                         family = "poisson",
                                         data,
                                         folds = 50,
                                         ratio = 0.8) {
  data <- data[, setdiff(colnames(data), no_modelling_vars)]
  
  if (family == "poisson") {
    data[, "AVG_SALES"] <- round(data[, "AVG_SALES"])
  }
  
  train_dfs <- list()
  test_dfs <- list()
  nrows <- nrow(data)
  train_size <- round(ratio * nrows)
  for (n in 1:folds) {
    filter <- as.numeric(row.names(data)) %in% sample(nrows, train_size)
    train_dfs[[n]] <- model.frame(formula, data[filter, ])
    test_dfs[[n]] <- model.frame(formula, data[!filter, ])
  }
  
  if (method == "NO_MODEL") {
    fits <- lapply(train_dfs, function(x) {
      lm(AVG_SALES ~ 1,
         data = x,
         distribution = family)
    })
    
    preds <- lapply(1:folds, function(x) {
      predict(fits[[x]], test_dfs[[x]], type = "response")
    })
    
    preds_mssr <- sapply(1:folds, function(x) {
      mean((preds[[x]] - test_dfs[[x]][, "AVG_SALES"]) ^ 2)
    })
    
    diagnostics_obj <- list(fits = fits,
                            preds = preds,
                            preds_mssr = preds_mssr)
  }
  
  if (method == "gbm") {
    fits <- lapply(train_dfs, function(x) {
      gbm(
        formula,
        data = x,
        distribution = family,
        n.trees = 200
      )
    })
    
    preds <- lapply(1:folds, function(x) {
      predict(fits[[x]], test_dfs[[x]], n.trees = 200, type = "response")
    })
    
    preds_mssr <- sapply(1:folds, function(x) {
      mean((preds[[x]] - test_dfs[[x]][, "AVG_SALES"]) ^ 2)
    })
    
    diagnostics_obj <- list(fits = fits,
                            preds = preds,
                            preds_mssr = preds_mssr)
  }
  
  if (method == "Random Forest") {
    fits <- lapply(train_dfs, function(x) {
      randomForest(formula,
                   data = x,
                   ntree = 200,
                   family = family)
    })
    
    preds <- lapply(1:folds, function(x) {
      predict(fits[[x]], test_dfs[[x]], ntree = 200, type = "response")
    })
    
    preds_mssr <- sapply(1:folds, function(x) {
      mean((preds[[x]] - test_dfs[[x]][, "AVG_SALES"]) ^ 2)
    })
    
    diagnostics_obj <- list(fits = fits,
                            preds = preds,
                            preds_mssr = preds_mssr)
  }
  return(diagnostics_obj)
}


create_mssr_comparisons_plot <- function(formula,
                                         no_modelling_vars = c("STORE_CODE",
                                                               "SALES_VOLATILITY",
                                                               "WEEKS_ONSALE",
                                                               "NOISY_POS",
                                                               "n_amenities"),
                                         family = "poisson",
                                         data,
                                         folds = 50,
                                         ratio = 0.8) {
  methods <- c("NO_MODEL", "gbm", "Random Forest")
  
  mssr <- lapply(methods,
                 function(x)
                   create_model_diagnostics_obj(
                     formula,
                     no_modelling_vars,
                     method = x,
                     family,
                     data,
                     folds,
                     ratio
                   )$preds_mssr)
  
  comparison_df <- data.frame(METHOD = rep(methods, each = folds),
                              MSSR = unlist(mssr))
  
  boxplot(MSSR ~ METHOD, data = comparison_df, main = "MSSR methods comparison")
  
}

modelling_df_preprocessing <- function(modelling_df,
                                       amenities_ls,
                                       cor_filter = T) {
  modelling_df <- modelling_df[complete.cases(modelling_df), ]
  if (cor_filter) {
    amenities_cor <-
      scale(abs(drop(
        cor(modelling_df[, "AVG_SALES"], modelling_df[, amenities_ls],
            method = "kendall")
      )))
    
    estimated_noise <-
      names(amenities_cor[, 1])[which(amenities_cor < 1)]
    modelling_df <-
      modelling_df[, setdiff(colnames(modelling_df), estimated_noise)]
  }
  return(modelling_df)
}
