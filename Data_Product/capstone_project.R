library(foreach)
library(iterators)
library(parallel)
library(doParallel)
library(bigmemory)
library(tidyverse)
library(multidplyr)
library(data.table)
library(tidytext)
library(tm)
library(NLP)
library(wordcloud)
library(gridExtra)

Log <- function(text, ...) {
    msg <- sprintf(paste0(as.character(Sys.time()), ": ", text), ...)
    
    cat(file = stderr(), msg, "\n")
    
    tryCatch({
        con <- socketConnection(host = "localhost", port = 22131, blocking = TRUE, server = FALSE, open = "r+")
        
        writeLines(msg, con)

        close(con)
    }, error = function(e) {
        ## Nothing
        # print(paste0("WARNING: ", e))
    })
}

loadingVar <- function(varName, override = FALSE) {
    fileName <- paste0(varName, ".rda")
    
    if(!file.exists(fileName)) {
        stop(paste0("ERROR: File not found: ", fileName))
    } else {
        if(override | !(exists(varName)))  load(fileName, envir = .GlobalEnv)
    }
}

## Split Corpus in a Training and Testing datasets
splitCorpus <- function(prop = .99) {
    if(!file.exists("training_US.rda") | !file.exists("testing_US.rda")) {
        
        if(!file.exists("Coursera-SwiftKey.zip") | !file.exists("final/en_US/en_US.twitter.txt") | !file.exists("final/en_US/en_US.blogs.txt") | !file.exists("final/en_US/en_US.news.txt")) {
            download.file("https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip", "Coursera-SwiftKey.zip", method = "curl")
            unzip("Coursera-SwiftKey.zip")
        }
        
        US_twitter <- readLines("final/en_US/en_US.twitter.txt", skipNul = TRUE)
        US_blogs <- readLines("final/en_US/en_US.blogs.txt", skipNul = TRUE)
        US_news <- readLines("final/en_US/en_US.news.txt", skipNul = TRUE)
        
        inTrain_twitter <- as.logical(rbinom(length(US_twitter), 1, prop))
        inTrain_blogs   <- as.logical(rbinom(length(US_blogs), 1, prop))
        inTrain_news    <- as.logical(rbinom(length(US_news), 1, prop))
        
        training_US <- data.frame(doc = c(
            US_twitter[inTrain_twitter],
            US_blogs[inTrain_blogs],
            US_news[inTrain_news]
        ), stringsAsFactors = FALSE)
        training_US$blocks <- rep(1:100, length.out = nrow(training_US))
        
        save(training_US, file = "training_US.rda")
        
        testing_US <- data.frame(doc = c(
            US_twitter[as.logical(-inTrain_twitter + 1)],
            US_blogs[as.logical(-inTrain_blogs + 1)],
            US_news[as.logical(-inTrain_news + 1)]
        ), stringsAsFactors = FALSE)
        
        save(testing_US, file = "testing_US.rda")
    } else {
        Log("INFO: Nothing to do. Data already split into a training and testing sets.")
    }    
}


## Process Training dataset
BigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
TrigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
FourgramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 4), paste, collapse = " "), use.names = FALSE)

combineLists <- function(a, b) {
    l <- list(a, b)
    keys <- unique(unlist(lapply(l, names)))
    
    setNames(lapply(keys, function(key) {
        l1 <- a[[key]]
        l2 <- b[[key]]
        
        rbind(l1, l2)
    }),
    keys)
}

cleaningCorpus <- function(corpus) {
    swearWords <- read.csv("swearWords.csv", header = FALSE, stringsAsFactors = FALSE)
    
    tm_map(corpus, removeNumbers) %>%
        tm_map(content_transformer(function(x) gsub("(f|ht)tp(s?)://(.*)[.][a-z]+[^\\s.]+", " ", x, perl = TRUE))) %>%
        tm_map(content_transformer(function(x) gsub("[@#][^\\s]+", " ", x, perl = TRUE))) %>%
        tm_map(content_transformer(tolower)) %>%
        tm_map(content_transformer(function(x) iconv(x, "latin1", "ASCII", sub = ""))) %>%
        tm_map(removeWords, swearWords$V1) %>%
        tm_map(removePunctuation, preserve_intra_word_dashes = TRUE) %>%
        tm_map(stripWhitespace)
}

splitConvertToId <- function(words, pos) {
    unigrams_dt_word[strsplit(words, " ")[[1]][pos], ]$idx
}

calculatingNgrams <- function(ratio = 1, cutOff = .9, filter_one_freq = TRUE) {
    if(!file.exists("unigrams_dt_word.rda") | !file.exists("unigrams_dt_idx.rda") | !file.exists("bigrams_bm.txt") | !file.exists("trigrams_bm.txt")) {
        cl <- detectCores() - 1
        
        loadingVar("training_US")
        
        results <- tryCatch({
            cluster <- makeCluster(cl)
            registerDoParallel(cluster)
            
            foreach(block = isplit(training_US, training_US[training_US$blocks %in% 1:(round(ratio * 101, 0)), ]$blocks),
                    .export = c('Log', 'cleaningCorpus', 'combineLists', 'BigramTokenizer', 'TrigramTokenizer', 'FourgramTokenizer'),
                    .packages = c('NLP', 'tm', 'dplyr', 'tidytext'),
                    .combine = 'combineLists') %dopar% {
                        
                        Log(paste0("INFO: block: ", block$key[[1]], ", size: ", utils:::format.object_size(object.size(block$value), "auto")))
                        
                        corpus <- VCorpus(VectorSource(block$value$doc))
                        cleanCorpus <- cleaningCorpus(corpus)
                        
                        ## unigrams
                        dtm_one <- DocumentTermMatrix(cleanCorpus, control = list(wordLengths = c(1, Inf)))
                        unigrams_tb <- tidy(dtm_one) %>% 
                            group_by(term) %>%
                            summarise(count = sum(count))
                        
                        ## bigrams
                        dtm_two <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = BigramTokenizer, wordLengths = c(1, Inf)))
                        bigrams_tb <- tidy(dtm_two) %>%
                            group_by(term) %>%
                            summarise(count = sum(count))
                        
                        ## trigrams
                        dtm_three <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = TrigramTokenizer, wordLengths = c(1, Inf)))
                        trigrams_tb <- tidy(dtm_three) %>%
                            group_by(term) %>%
                            summarise(count = sum(count))
                        
                        list(unigrams_tb = unigrams_tb, bigrams_tb = bigrams_tb, trigrams_tb = trigrams_tb)
                    }
        }, error = function(e) {
            Log(paste0("ERROR: ", e))
        }, finally = {
            Log("INFO: finally handler called")
            
            stopCluster(cluster)
            registerDoSEQ()
        })
        
        percentile <- function(data, p = .5) {
            cumul_nb <- 0
            i <- 1
            
            while(cumul_nb < (p * sum(data$count))) {
                cumul_nb <- cumul_nb + data[i, ]$count
                i <- i + 1
            }
            data[i - 1, ]$count
        }
        
        # Unigrams
        Log("INFO: Treating unigrams tb")
        unigrams <- results[["unigrams_tb"]] %>%
            group_by(term) %>%
            summarise(count = sum(count)) %>%
            arrange(desc(count))
        
        nb <- percentile(unigrams, cutOff)
        
        unigrams <- unigrams %>%
            filter(count > nb) %>%
            mutate(prob = count / sum(count))
        
        unigrams$term <- as.factor(unigrams$term)
        
        unigrams_dt_word <- data.table(
            idx = as.numeric(unigrams$term),
            word = as.character(unigrams$term),
            count = unigrams$count,
            prob = unigrams$prob)
        setkey(unigrams_dt_word, word)
        save(unigrams_dt_word, file = "unigrams_dt_word.rda")
        
        unigrams_dt_idx <- data.table(
            idx = as.numeric(unigrams$term),
            word = as.character(unigrams$term),
            count = unigrams$count,
            prob = unigrams$prob)
        setkey(unigrams_dt_idx, idx)
        save(unigrams_dt_idx, file = "unigrams_dt_idx.rda")
        
        rm(unigrams)
        
        
        # Bigrams
        Log("INFO: Treating bigrams tb")
        bigrams <- results[["bigrams_tb"]] %>%
            group_by(term) %>%
            summarise(count = sum(count))
        
        if(filter_one_freq) {
            bigrams <- filter(bigrams, count > 1)    
        }
        
        group <- rep(1:cl, length.out = nrow(bigrams))
        
        bigrams <- bind_cols(tibble(group), bigrams) %>%
            partition(group, cluster = create_cluster(cl)) %>%
            cluster_library("tidyverse") %>%
            cluster_library("data.table") %>%
            cluster_assign_value("unigrams_dt_idx", unigrams_dt_idx) %>%
            cluster_assign_value("unigrams_dt_word", unigrams_dt_word) %>%
            cluster_assign_value("splitConvertToId", splitConvertToId) %>%
            mutate(
                Wi_1 = map_dbl(.x = term, ~ splitConvertToId(.x, 1)),
                Wi   = map_dbl(.x = term, ~ splitConvertToId(.x, 2))
            ) %>%
            filter(Wi_1 %in% unigrams_dt_idx$idx) %>%
            filter(Wi   %in% unigrams_dt_idx$idx) %>%
            collect() %>%
            ungroup() %>%
            select(Wi_1, Wi, count) %>%
            arrange(Wi_1, Wi)
        
        write.big.matrix(as.big.matrix(as.data.frame(bigrams), type = "integer"), "bigrams_bm.txt")
        rm(bigrams)
        
        
        # Trigrams
        Log("INFO: Treating trigrams tb")
        trigrams <- results[["trigrams_tb"]] %>%
            group_by(term) %>%
            summarise(count = sum(count))
        
        if(filter_one_freq) {
            trigrams <- filter(trigrams, count > 1)    
        }
        
        rm(results)
        group <- rep(1:cl, length.out = nrow(trigrams))
        
        trigrams <- bind_cols(tibble(group), trigrams) %>%
            partition(group, cluster = create_cluster(cl)) %>%
            cluster_library("tidyverse") %>%
            cluster_library("data.table") %>%
            cluster_assign_value("unigrams_dt_idx", unigrams_dt_idx) %>%
            cluster_assign_value("unigrams_dt_word", unigrams_dt_word) %>%
            cluster_assign_value("splitConvertToId", splitConvertToId) %>%
            mutate(
                Wi_2 = map_dbl(.x = term, ~ splitConvertToId(.x, 1)),
                Wi_1 = map_dbl(.x = term, ~ splitConvertToId(.x, 2)),
                Wi   = map_dbl(.x = term, ~ splitConvertToId(.x, 3))
            ) %>%
            filter(Wi_2 %in% unigrams_dt_idx$idx) %>%
            filter(Wi_1 %in% unigrams_dt_idx$idx) %>%
            filter(Wi   %in% unigrams_dt_idx$idx) %>%
            collect() %>%
            ungroup() %>%
            select(Wi_2, Wi_1, Wi, count) %>%
            arrange(Wi_2, Wi_1, Wi)
        
        write.big.matrix(as.big.matrix(as.data.frame(trigrams), type = "integer"), "trigrams_bm.txt")
        rm(trigrams)
        
        rm(unigrams_dt_idx)
        rm(unigrams_dt_word)
    } else {
        Log("INFO: Nothing to do. Data already cleaned (rda files exists).")
    }
}


## Statistics about the model
tableStats <- function() {
    loadingVar("unigrams_dt_word")
    bigrams_desc <- dget("bigrams.desc")
    bigrams <- attach.big.matrix(bigrams_desc)
    trigrams_desc <- dget("trigrams.desc")
    trigrams <- attach.big.matrix(trigrams_desc)
    
    vocabulary <- nrow(unigrams_dt_word)
    
    stats <- matrix(c(
        vocabulary,
        vocabulary,
        vocabulary / vocabulary,
        
        nrow(bigrams),
        vocabulary^2,
        nrow(bigrams) / vocabulary^2,
        
        nrow(trigrams),
        vocabulary^3,
        nrow(trigrams) / vocabulary^3
    ), ncol = 3, byrow = FALSE, dimnames = list(c("observed", "V^N", "proportion"), c("unigrams", "bigrams", "trigrams")))
    
    print(stats)
}

plotStats <- function() {
    loadingVar("unigrams_dt_word")
    loadingVar("unigrams_dt_idx")
    
    unigrams <- unigrams_dt_word %>%
        arrange(desc(count)) %>%
        select(word, count)
    
    bigrams_desc <- dget("bigrams.desc")
    bigrams <- attach.big.matrix(bigrams_desc)
    trigrams_desc <- dget("trigrams.desc")
    trigrams <- attach.big.matrix(trigrams_desc)
    
    plot1 <- ggplot(unigrams[1:20, ], aes(reorder(word, desc(count)), count)) +
        geom_bar(stat = "identity") +
        ggtitle("Top 20 more recurrent unigrams") +
        labs(x = "Terms", y = "Count") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1), plot.title = element_text(hjust = 0.5))
    
    bigrams <- as.data.frame(as.matrix(bigrams)) %>%
        arrange(desc(count)) %>%
        mutate(
            W1 = unigrams_dt_idx[Wi_1]$word,
            W2 = unigrams_dt_idx[Wi]$word
        ) %>%
        mutate(term = paste(W1, W2)) 
    plot2 <- ggplot(bigrams[1:20, ], aes(reorder(term, desc(count)), count)) +
        geom_bar(stat = "identity") +
        ggtitle("Top 20 more recurrent bigrams") +
        labs(x = "Terms", y = "Count") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1), plot.title = element_text(hjust = 0.5))
    
    trigrams <- as.data.frame(as.matrix(trigrams)) %>%
        arrange(desc(count)) %>%
        mutate(
            W1 = unigrams_dt_idx[Wi_2]$word,
            W2 = unigrams_dt_idx[Wi_1]$word,
            W3 = unigrams_dt_idx[Wi]$word
        ) %>%
        mutate(term = paste(W1, W2, W3)) 
    plot3 <- ggplot(trigrams[1:20, ], aes(reorder(term, desc(count)), count)) +
        geom_bar(stat = "identity") +
        ggtitle("Top 20 more recurrent trigrams") +
        labs(x = "Terms", y = "Count") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1), plot.title = element_text(hjust = 0.5))
    
    grid.arrange(plot1, plot2, plot3)
    
    
    wordcloud(unigrams$word, unigrams$count, max.words = 250, colors = brewer.pal(6, "Dark2"))
    wordcloud(bigrams$term,  bigrams$count,  max.words = 250, colors = brewer.pal(6, "Dark2"))
    wordcloud(trigrams$term, trigrams$count, max.words = 250, colors = brewer.pal(6, "Dark2"))
}


## Assigning Probabilities using the Maximum Likelihood Estimate
### Katz back-off algorithm
calcQBi <- function(WWi, WWi_1, AWi_1, BWi_1, alphaWi_1, d = .5) {
    if(WWi %in% AWi_1) {
        qBi <- (bigrams[mwhich(bigrams, c("Wi_1", "Wi"), list(WWi_1, WWi), list("eq", "eq"), "AND"), "count"] - d) / 
            sum(bigrams[mwhich(bigrams, "Wi_1", WWi_1, "eq"), "count"])
    } else {
        qBi <- alphaWi_1 * unigrams_dt_idx[WWi, ]$prob / sum(unigrams_dt_idx[.(BWi_1)]$prob)
    }
    
    qBi
}

calcQTri <- function(WWi, WWi_1, WWi_2, AWi_2Wi_1, BWi_2Wi_1, alphaWi_2Wi_1, denom, d = .5) {
    if(is.null(AWi_2Wi_1)) {
        AWi_2Wi_1 <- unigrams_dt_idx[.(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"), "Wi"])]$idx
    }
    if(is.null(BWi_2Wi_1)) {
        BWi_2Wi_1 <- unigrams_dt_idx[!(.(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"), "Wi"]))]$idx
    }
    
    if(WWi %in% AWi_2Wi_1) {
        qTri <- (trigrams[mwhich(trigrams, c("Wi_2", "Wi_1", "Wi"), list(WWi_2, WWi_1, WWi), list("eq", "eq", "eq"), "AND"), "count"] - d) / 
            sum(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"), "count"])
    } else {
        if(is.null(alphaWi_2Wi_1)) {
            alphaWi_2Wi_1 <- 1 - sum(
                (trigrams[intersect(
                    mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"),
                    which(trigrams[, "Wi"] %in% AWi_2Wi_1)
                ), "count"] - d) /
                    sum(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"), "count"])
            )
        }
        
        AWi_1 <- unigrams_dt_idx[.(bigrams[mwhich(bigrams, "Wi_1", WWi_1, "eq"), "Wi"])]$idx
        BWi_1 <- unigrams_dt_idx[!(.(bigrams[mwhich(bigrams, "Wi_1", WWi_1, "eq"), "Wi"]))]$idx
        
        alphaWi_1 <- 1 - sum(
            (bigrams[intersect(
                mwhich(bigrams, "Wi_1", WWi_1, "eq", "AND"),
                which(bigrams[, "Wi"] %in% AWi_1)
            ), "count"] - d) /
            sum(bigrams[mwhich(bigrams, "Wi_1", WWi_1, "eq"), "count"])
        )
        
        if(is.null(denom)) {
            denom <- foreach(word = BWi_2Wi_1, 
                             .export = c('calcQBi', 'unigrams_dt_idx', 'bigrams_desc'),
                             .combine = '+') %do% {
                                 
                require(bigmemory)
                
                bigrams <- attach.big.matrix(bigrams_desc)
                calcQBi(word, WWi_1, AWi_1, BWi_1, alphaWi_1)
            }
        }
        
        qTri <- alphaWi_2Wi_1 * calcQBi(WWi, WWi_1, AWi_1, BWi_1, alphaWi_1) / denom
    }
    
    qTri
}

### Stupid back-off algorithm
stupidCalcQBi <- function(WWi, WWi_1, a = .4) {
    AWi_1 <- unigrams_dt_idx[.(bigrams[mwhich(bigrams, "Wi_1", WWi_1, "eq"), "Wi"])]$idx
    
    if(WWi %in% AWi_1) {
        qBi <- bigrams[mwhich(bigrams, c("Wi_1", "Wi"), list(WWi_1, WWi), list("eq", "eq")), "count"] / 
               sum(bigrams[mwhich(bigrams, "Wi_1", WWi_1, "eq"), "count"])
    } else {
        if(!is.na(WWi)) {
            qBi <- a * unigrams_dt_idx[WWi, ]$prob
        } else {
            qBi <- 1 / nrow(unigrams_dt_idx)
        }
    }
    
    qBi
}

stupidCalcQTri <- function(WWi, WWi_1, WWi_2, AWi_2Wi_1, a = .4) {
    if(is.null(AWi_2Wi_1)) {
        AWi_2Wi_1 <- unigrams_dt_idx[.(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"), "Wi"])]$idx
    }
    
    if(WWi %in% AWi_2Wi_1) {
        qTri <- trigrams[mwhich(trigrams, c("Wi_2", "Wi_1", "Wi"), list(WWi_2, WWi_1, WWi), list("eq", "eq", "eq"), "AND"), "count"] /
                sum(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(WWi_2, WWi_1), list("eq", "eq"), "AND"), "count"])
    } else {
        qTri <- a * stupidCalcQBi(WWi, WWi_1)
    }
    
    qTri
}


## Estimation of Accuracy
### number of hits (among top N predicted words) vs total number of tests
accuracyNChoices <- function(ratio = .001, N = 3) {
    if(!(is.numeric(N) & N > 0 & N <= 5)) {
        stop("N must be a number > 0 and <= 5.")
    }
    
    Log("INFO: Loading vars...")
    loadingVar("testing_US")
    loadingVar("unigrams_dt_idx")
    loadingVar("unigrams_dt_word")
    bigrams_desc <- dget("bigrams.desc")
    bigrams <- attach.big.matrix(bigrams_desc)
    trigrams_desc <- dget("trigrams.desc")
    trigrams <- attach.big.matrix(trigrams_desc)
    
    corpus <- VCorpus(VectorSource(testing_US[1:50, ]$doc)) # round(nrow(testing_US) * ratio, 0)
    Log("INFO: Cleaning Corpus...")
    cleanCorpus <- cleaningCorpus(corpus)
    
    Log("INFO: Extracting unigrams and trigrams...")
    dtm_three <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = TrigramTokenizer, wordLengths = c(1, Inf)))
    
    trigrams_tb <- tidy(dtm_three)
    nb_rows <- nrow(trigrams_tb)
    
    Log(paste0("INFO: Counting hits..."))
    trigrams_tb <- trigrams_tb %>%
        mutate(
            Wi_2 = map_dbl(.x = term, ~ splitConvertToId(.x, 1)),
            Wi_1 = map_dbl(.x = term, ~ splitConvertToId(.x, 2)),
            Wi   = map_dbl(.x = term, ~ splitConvertToId(.x, 3))
        ) %>%
        filter(!is.na(Wi_2)) %>%
        filter(!is.na(Wi_1)) %>%
        filter(!is.na(Wi)) %>%
        rowwise() %>%
        do(data.frame(Wi = .$Wi, pred = stupidPredictNextWord(paste(unigrams_dt_idx[.$Wi_2]$word, unigrams_dt_idx[.$Wi_1]$word))))  %>%
        top_n(N, pred.prob) %>%
        summarise(match = sum(Wi == pred.word_idx)) %>%
        ungroup %>% 
        summarise(hits = sum(match != 0))
    
    Log(paste0("INFO: hits: ", trigrams_tb$hits, ", nb_rows: ", nb_rows))
    accuracy <- trigrams_tb$hits / nb_rows
    
    Log(paste0("INFO: accuracy: ", round(100 * accuracy, 2), "%"))
    
    accuracy
}

### Perplexity (deprecated)
perplexity <- function(ratio = .001) {
    Log("Loading vars...")
    loadingVar("testing_US")
    loadingVar("unigrams_dt_word")
    loadingVar("unigrams_dt_idx")
    bigrams_desc <- dget("bigrams.desc")
    bigrams <- attach.big.matrix(bigrams_desc)
    trigrams_desc <- dget("trigrams.desc")
    trigrams <- attach.big.matrix(trigrams_desc)
    
    cluster <- makeCluster(detectCores() - 1)
    registerDoParallel(cluster)
    
    corpus <- VCorpus(VectorSource(testing_US[1:round(nrow(testing_US) * ratio, 0), ]$doc))
    Log("Cleaning Corpus...")
    cleanCorpus <- cleaningCorpus(corpus)
    
    Log("Extracting unigrams and trigrams...")
    dtm_one   <- DocumentTermMatrix(cleanCorpus, control = list(wordLengths = c(1, Inf)))
    dtm_three <- DocumentTermMatrix(cleanCorpus, control = list(tokenize = TrigramTokenizer, wordLengths = c(1, Inf)))
    
    unigrams_tb <- tidy(dtm_one)
    trigrams_tb <- tidy(dtm_three)
    
    tryCatch({
        Log("Calculating probabilities...")
        prob_Si <- foreach(block = isplit(trigrams_tb, trigrams_tb$document),
                           .export = c('Log', 'stupidCalcQTri', 'splitConvertToId', 'stupidCalcQBi', 'unigrams_dt_idx', 'unigrams_dt_word'),
                           .packages = c("dplyr", "foreach", "tidyverse", "data.table"),
                           .combine = 'rbind') %dopar% {

            Log(paste0("INFO: block: ", block$key[[1]], ", size: ", utils:::format.object_size(object.size(block$value), "auto")))
                               
            require(bigmemory)
               
            bigrams <- attach.big.matrix(bigrams_desc)
            trigrams <- attach.big.matrix(trigrams_desc)
            
            block$value %>%
                mutate(
                    Wi_2 = map_dbl(.x = term, ~ splitConvertToId(.x, 1)),
                    Wi_1 = map_dbl(.x = term, ~ splitConvertToId(.x, 2)),
                    Wi   = map_dbl(.x = term, ~ splitConvertToId(.x, 3))
                ) %>%
                group_by(Wi_2, Wi_1, Wi, term) %>%
                summarise(count = sum(count)) %>%
                rowwise() %>%
                do(data.frame(prob = stupidCalcQTri(.$Wi, .$Wi_1, .$Wi_2, NULL))) %>%  ## calcQTri(.$Wi, .$Wi_1, .$Wi_2, NULL, NULL, NULL, NULL)
                ungroup() %>%
                summarise(prob = prod(prob))
        }
        
        M <- length(unique(unigrams_tb$term))
        l <- (1 / M) * sum(log(prob_Si$prob, base = 2))
        perplexity <- 2^(-l)

        Log(paste0("INFO: perplexity: ", perplexity))

    }, error = function(e) {
        print(paste0("ERROR: ", e))
    }, finally = {
        Log("INFO: finally handler called")

        stopCluster(cluster)
        registerDoSEQ()
    })
}


## Predicting Next Word
### Using Katz back-off model
predictNextWord <- function(words, resultNb = 10, incStopWords = TRUE, d = .5) {
    loadingVar("unigrams_dt_idx")
    loadingVar("unigrams_dt_word")
    bigrams_desc <- dget("bigrams.desc")
    bigrams <- attach.big.matrix(bigrams_desc)
    trigrams_desc <- dget("trigrams.desc")
    trigrams <- attach.big.matrix(trigrams_desc)
    
    words <- unlist(strsplit(tolower(words), " "))
    nb_words <- length(words)
    
    if(nb_words >= 2) {
        words <- words[(nb_words - 1):nb_words]
        
        cluster <- makeCluster(detectCores() - 1)
        registerDoParallel(cluster)
        
        w1_idx <- unigrams_dt_word[words[1], ]$idx
        w2_idx <- unigrams_dt_word[words[2], ]$idx
        
        tested_words <- merge(data.frame(w1_idx, w2_idx), unigrams_dt_idx$idx, all = TRUE)
        names(tested_words) <- c("V1", "V2", "V3")
        tested_words$blocks <- rep(1:100, length.out = nrow(tested_words))
        
        tryCatch({
            ## For trigrams
            Log("Identify AWi_2Wi_1, BWi_2Wi_1 and calculate alphaWi_2Wi_1 for trigrams")
            
            AWi_2Wi_1 <- unigrams_dt_idx[.(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(w1_idx, w2_idx), list("eq", "eq"), "AND"), "Wi"])]$idx
            BWi_2Wi_1 <- unigrams_dt_idx[!(.(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(w1_idx, w2_idx), list("eq", "eq"), "AND"), "Wi"]))]$idx
            
            alphaWi_2Wi_1 <- 1 - sum(
                (trigrams[intersect(
                    mwhich(trigrams, c("Wi_2", "Wi_1"), list(w1_idx, w2_idx), list("eq", "eq"), "AND"),
                    which(trigrams[, "Wi"] %in% AWi_2Wi_1)
                ), "count"] - d) /
                sum(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(w1_idx, w2_idx), list("eq", "eq"), "AND"), "count"])
            )
            
            ## For bigrams
            Log("Identify AWi_1, BWi_1 and calculate alphaWi_1 for bigrams")
            AWi_1 <- unigrams_dt_idx[.(bigrams[mwhich(bigrams, "Wi_1", w2_idx, "eq"), "Wi"])]$idx
            BWi_1 <- unigrams_dt_idx[!(.(bigrams[mwhich(bigrams, "Wi_1", w2_idx, "eq"), "Wi"]))]$idx
            
            alphaWi_1 <- 1 - sum(
                (bigrams[intersect(
                    mwhich(bigrams, "Wi_1", w2_idx, "eq"),
                    which(bigrams[, "Wi"] %in% AWi_1)
                ), "count"] - d) /
                sum(bigrams[mwhich(bigrams, "Wi_1", w2_idx, "eq"), "count"])
            )
            
            ## Calculate bigrams adjusted prob
            Log("Calculate bigrams adjusted prob")
            denom <- foreach(word = BWi_2Wi_1, 
                             .export = c('Log', 'calcQBi', 'unigrams_dt_idx'),
                             .packages = c("data.table"),
                             .combine = '+') %dopar% {
                                 
                require(bigmemory)
                
                bigrams <- attach.big.matrix(bigrams_desc)
                calcQBi(word, w2_idx, AWi_1, BWi_1, alphaWi_1, d)
            }
            
            ## Calculate trigrams' prob
            Log("Calculate trigrams' prob")
            results <- foreach(block = isplit(tested_words, tested_words$blocks), 
                               .export = c('Log', 'calcQTri', 'calcQBi', 'unigrams_dt_idx'),
                               .packages = c("data.table"),
                               .combine = 'rbind') %dopar% {
                                   
                require(bigmemory)
                                   
                bigrams <- attach.big.matrix(bigrams_desc)
                trigrams <- attach.big.matrix(trigrams_desc)
                
                nb <- dim(block$value)[1]
                   
                result <- data.frame(
                    word_idx = numeric(nb),
                    word = character(nb),
                    prob = numeric(nb),
                    stringsAsFactors = FALSE
                )
                   
                for (i in 1:nb) {
                    value <- block$value
                    result[i, ]$prob <- calcQTri(value[i, ]$V3, value[i, ]$V2, value[i, ]$V1, AWi_2Wi_1, BWi_2Wi_1, alphaWi_2Wi_1, denom, d)
                    result[i, ]$word <- unigrams_dt_idx[value[i, ]$V3, ]$word
                    result[i, ]$word_idx <- value[i, ]$V3
                }
                result
            }
            
            if(!incStopWords) {
                stopwords <- unlist(strsplit("a, about, above, across, after, again, against, all, almost, alone, along, already, also, although, always, am, among, an, and, another, any, anybody, anyone, anything, anywhere, are, area, areas, aren't, around, as, ask, asked, asking, asks, at, away, b, back, backed, backing, backs, be, became, because, become, becomes, been, before, began, behind, being, beings, below, best, better, between, big, both, but, by, c, came, can, cannot, can't, case, cases, certain, certainly, clear, clearly, come, could, couldn't, d, did, didn't, differ, different, differently, do, does, doesn't, doing, done, don't, down, downed, downing, downs, during, e, each, early, either, end, ended, ending, ends, enough, even, evenly, ever, every, everybody, everyone, everything, everywhere, f, face, faces, fact, facts, far, felt, few, find, finds, first, for, four, from, full, fully, further, furthered, furthering, furthers, g, gave, general, generally, get, gets, give, given, gives, go, going, good, goods, got, great, greater, greatest, group, grouped, grouping, groups, h, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, her, here, here's, hers, herself, he's, high, higher, highest, him, himself, his, how, however, how's, i, i'd, if, i'll, i'm, important, in, interest, interested, interesting, interests, into, is, isn't, it, its, it's, itself, i've, j, just, k, keep, keeps, kind, knew, know, known, knows, l, large, largely, last, later, latest, least, less, let, lets, let's, like, likely, long, longer, longest, m, made, make, making, man, many, may, me, member, members, men, might, more, most, mostly, mr, mrs, much, must, mustn't, my, myself, n, necessary, need, needed, needing, needs, never, new, newer, newest, next, no, nobody, non, noone, nor, not, nothing, now, nowhere, number, numbers, o, of, off, often, old, older, oldest, on, once, one, only, open, opened, opening, opens, or, order, ordered, ordering, orders, other, others, ought, our, ours, ourselves, out, over, own, p, part, parted, parting, parts, per, perhaps, place, places, point, pointed, pointing, points, possible, present, presented, presenting, presents, problem, problems, put, puts, q, quite, r, rather, really, right, room, rooms, s, said, same, saw, say, says, second, seconds, see, seem, seemed, seeming, seems, sees, several, shall, shan't, she, she'd, she'll, she's, should, shouldn't, show, showed, showing, shows, side, sides, since, small, smaller, smallest, so, some, somebody, someone, something, somewhere, state, states, still, such, sure, t, take, taken, than, that, that's, the, their, theirs, them, themselves, then, there, therefore, there's, these, they, they'd, they'll, they're, they've, thing, things, think, thinks, this, those, though, thought, thoughts, three, through, thus, to, today, together, too, took, toward, turn, turned, turning, turns, two, u, under, until, up, upon, us, use, used, uses, v, very, w, want, wanted, wanting, wants, was, wasn't, way, ways, we, we'd, well, we'll, wells, went, were, we're, weren't, we've, what, what's, when, when's, where, where's, whether, which, while, who, whole, whom, who's, whose, why, why's, will, with, within, without, won't, work, worked, working, works, would, wouldn't, x, y, year, years, yes, yet, you, you'd, you'll, young, younger, youngest, your, you're, yours, yourself, yourselves, you've, z", ", "))
                results <- results %>%
                    filter(!(word %in% stopwords))
            }
            
            
            ## Displaying top 10 most probable words with corresponding probabilities
            Log("Displaying top 10 most probable words with corresponding probabilities")
            results %>%
                arrange(desc(prob)) %>%
                top_n(resultNb, prob)
            
        }, error = function(e) {
            Log(paste0("ERROR: ", e))
        }, finally = {
            Log("INFO: finally handler called")
            
            stopCluster(cluster)
            registerDoSEQ()
        })
    } else {
        stop("The provided string should be  at least two words long.")
    }
}

### Using Stupid back-off model
stupidPredictNextWord <- function(words, resultNb = 10, incStopWords = TRUE) {
    
    loadingVar("unigrams_dt_idx")
    loadingVar("unigrams_dt_word")
    bigrams_desc <- dget("bigrams.desc")
    bigrams <- attach.big.matrix(bigrams_desc)
    trigrams_desc <- dget("trigrams.desc")
    trigrams <- attach.big.matrix(trigrams_desc)
    
    words <- unlist(strsplit(tolower(words), " "))
    nb_words <- length(words)
    
    if(nb_words >= 2) {
        words <- words[(nb_words - 1):nb_words]
        
        cluster <- makeCluster(detectCores() - 1)
        registerDoParallel(cluster)
        
        w1_idx <- unigrams_dt_word[words[1], ]$idx
        w2_idx <- unigrams_dt_word[words[2], ]$idx
        
        tested_words <- merge(data.frame(w1_idx, w2_idx), unigrams_dt_idx$idx, all = TRUE)
        names(tested_words) <- c("V1", "V2", "V3")
        tested_words$blocks <- rep(1:100, length.out = nrow(tested_words))
        
        tryCatch({
            ## For trigrams
            Log("Identify AWi_2Wi_1 for trigrams")
            AWi_2Wi_1 <- unigrams_dt_idx[.(trigrams[mwhich(trigrams, c("Wi_2", "Wi_1"), list(w1_idx, w2_idx), list("eq", "eq"), "AND"), "Wi"])]$idx
            
            ## Calculate trigrams' prob
            Log("Calculate trigrams' prob")
            results <- foreach(block = isplit(tested_words, tested_words$blocks), 
                               .export = c('Log', 'stupidCalcQTri', 'stupidCalcQBi', 'unigrams_dt_idx'),
                               .packages = c("data.table"),
                               .combine = 'rbind') %dopar% {
                                   
                require(bigmemory)
                   
                bigrams <- attach.big.matrix(bigrams_desc)
                trigrams <- attach.big.matrix(trigrams_desc)
                   
                nb <- dim(block$value)[1]
                   
                result <- data.frame(
                    word_idx = numeric(nb),
                    word = character(nb),
                    prob = numeric(nb),
                    stringsAsFactors = FALSE
                )
                   
                for (i in 1:nb) {
                    value <- block$value
                    result[i, ]$prob <- stupidCalcQTri(value[i, ]$V3, value[i, ]$V2, value[i, ]$V1, AWi_2Wi_1)
                    result[i, ]$word <- unigrams_dt_idx[value[i, ]$V3, ]$word
                    result[i, ]$word_idx <- value[i, ]$V3
                }
                result
            }
            
            if(!incStopWords) {
                stopwords <- unlist(strsplit("a, about, above, across, after, again, against, all, almost, alone, along, already, also, although, always, am, among, an, and, another, any, anybody, anyone, anything, anywhere, are, area, areas, aren't, around, as, ask, asked, asking, asks, at, away, b, back, backed, backing, backs, be, became, because, become, becomes, been, before, began, behind, being, beings, below, best, better, between, big, both, but, by, c, came, can, cannot, can't, case, cases, certain, certainly, clear, clearly, come, could, couldn't, d, did, didn't, differ, different, differently, do, does, doesn't, doing, done, don't, down, downed, downing, downs, during, e, each, early, either, end, ended, ending, ends, enough, even, evenly, ever, every, everybody, everyone, everything, everywhere, f, face, faces, fact, facts, far, felt, few, find, finds, first, for, four, from, full, fully, further, furthered, furthering, furthers, g, gave, general, generally, get, gets, give, given, gives, go, going, good, goods, got, great, greater, greatest, group, grouped, grouping, groups, h, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, her, here, here's, hers, herself, he's, high, higher, highest, him, himself, his, how, however, how's, i, i'd, if, i'll, i'm, important, in, interest, interested, interesting, interests, into, is, isn't, it, its, it's, itself, i've, j, just, k, keep, keeps, kind, knew, know, known, knows, l, large, largely, last, later, latest, least, less, let, lets, let's, like, likely, long, longer, longest, m, made, make, making, man, many, may, me, member, members, men, might, more, most, mostly, mr, mrs, much, must, mustn't, my, myself, n, necessary, need, needed, needing, needs, never, new, newer, newest, next, no, nobody, non, noone, nor, not, nothing, now, nowhere, number, numbers, o, of, off, often, old, older, oldest, on, once, one, only, open, opened, opening, opens, or, order, ordered, ordering, orders, other, others, ought, our, ours, ourselves, out, over, own, p, part, parted, parting, parts, per, perhaps, place, places, point, pointed, pointing, points, possible, present, presented, presenting, presents, problem, problems, put, puts, q, quite, r, rather, really, right, room, rooms, s, said, same, saw, say, says, second, seconds, see, seem, seemed, seeming, seems, sees, several, shall, shan't, she, she'd, she'll, she's, should, shouldn't, show, showed, showing, shows, side, sides, since, small, smaller, smallest, so, some, somebody, someone, something, somewhere, state, states, still, such, sure, t, take, taken, than, that, that's, the, their, theirs, them, themselves, then, there, therefore, there's, these, they, they'd, they'll, they're, they've, thing, things, think, thinks, this, those, though, thought, thoughts, three, through, thus, to, today, together, too, took, toward, turn, turned, turning, turns, two, u, under, until, up, upon, us, use, used, uses, v, very, w, want, wanted, wanting, wants, was, wasn't, way, ways, we, we'd, well, we'll, wells, went, were, we're, weren't, we've, what, what's, when, when's, where, where's, whether, which, while, who, whole, whom, who's, whose, why, why's, will, with, within, without, won't, work, worked, working, works, would, wouldn't, x, y, year, years, yes, yet, you, you'd, you'll, young, younger, youngest, your, you're, yours, yourself, yourselves, you've, z", ", "))
                results <- results %>%
                    filter(!(word %in% stopwords))
            }
            
            ## Displaying top 10 most probable words with corresponding probabilities
            Log("Displaying top 10 most probable words with corresponding probabilities")
            results %>%
                arrange(desc(prob)) %>%
                top_n(resultNb, prob)
            
        }, error = function(e) {
            Log(paste0("ERROR: ", e))
        }, finally = {
            Log("INFO: finally handler called")
            
            stopCluster(cluster)
            registerDoSEQ()
        })
    } else {
        stop("The provided string should be at least two words long.")
    }
}


## Optimize Language Model
optimizeModel <- function(sampling = c(.20, .25, .30, 1), cutoff = c(.8, .85, .9), filter_one_freq = c(TRUE, FALSE)) {
    
    model_params <- expand.grid(
        sampling = sampling, 
        cutoff = cutoff, 
        filter_one_freq = filter_one_freq
    )
    
    nb <- nrow(model_params)
    
    results <- data.frame(
        code = character(nb),
        label1 = character(nb),
        label2 = character(nb),
        label3 = character(nb),
        unigrams = numeric(nb),
        bigrams = numeric(nb),
        trigrams = numeric(nb),
        stupidPred = numeric(nb),
        pred = numeric(nb),
        accuracy = numeric(nb),
        stringsAsFactors = FALSE
    )
    
    for(i in 1:nrow(model_params)) {
        rm(unigrams_dt_word)
        rm(unigrams_dt_idx)
        rm(bigrams)
        rm(trigrams)
        
        file.remove("unigrams_dt_word.rda")
        file.remove("unigrams_dt_idx.rda")
        file.remove("bigrams_bm.txt")
        file.remove("bigrams.desc")
        file.remove("bigrams.bin")
        file.remove("trigrams_bm.txt")
        file.remove("trigrams.desc")
        file.remove("trigrams.bin")
        
        results[i, "code"] <- paste(
            round(100 * model_params[i, ]$sampling, 0),
            round(100 * model_params[i, ]$cutoff, 0),
            model_params[i, ]$filter_one_freq,
            sep = "_")
        results[i, "label1"] <- paste(
            round(100 * model_params[i, ]$sampling, 0),
            round(100 * model_params[i, ]$cutoff, 0))
        results[i, "label2"] <- paste(
            round(100 * model_params[i, ]$sampling, 0),
            model_params[i, ]$filter_one_freq)
        results[i, "label3"] <- paste(
            round(100 * model_params[i, ]$cutoff, 0),
            model_params[i, ]$filter_one_freq)
        
        
        Log("calculatingNgrams...")
        calculatingNgrams(model_params[i, ]$sampling, model_params[i, ]$cutoff, model_params[i, ]$filter_one_freq)
        gc()
        
        load("unigrams_dt_word.rda")
        
        bigrams  <- read.big.matrix("bigrams_bm.txt",  sep = ",", type = "integer", shared = TRUE, col.names = c("Wi_1", "Wi", "count"), descriptorfile = "bigrams.desc", backingfile = "bigrams.bin")
        trigrams <- read.big.matrix("trigrams_bm.txt", sep = ",", type = "integer", shared = TRUE, col.names = c("Wi_2", "Wi_1", "Wi", "count"), descriptorfile = "trigrams.desc", backingfile = "trigrams.bin")
        
        results[i, ]$unigrams <- nrow(unigrams_dt_word)
        results[i, ]$bigrams <- nrow(bigrams)
        results[i, ]$trigrams <- nrow(trigrams)
        
        results[i, ]$stupidPred <- system.time(stupidPredictNextWord("one of"))[["elapsed"]]
        results[i, ]$pred <- system.time(predictNextWord("one of"))[["elapsed"]]
        
        Log("accuracyNChoices...")
        results[i, ]$accuracy <- accuracyNChoices()
        
        save(results, file = "results.rda")
        
        gc()
    }
}