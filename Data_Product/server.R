library(shiny)
source("capstone_project.R")

shinyServer(function(input, output) {
    predictions <- reactive({
        pred <- NULL
        if(input$algo == "katz") {
            pred <- predictNextWord(input$words, as.numeric(input$resultNb))[, c("word", "prob")]
        } else {
            pred <- stupidPredictNextWord(input$words, as.numeric(input$resultNb))[, c("word", "prob")]
        }
        pred$prob <- paste0(round(100 * pred$prob, 2), "%")
        names(pred) <- c("Word", "Probability")
        
        pred
    })
    
    output$prediction <- renderDataTable(
        predictions(),
        options = list(
            pageLength = 10,
            lengthMenu = FALSE,
            searching = FALSE,
            ordering = FALSE,
            info = FALSE,
            paging = FALSE
        )
    )
})