library(shiny)
source("capstone_project.R")

shinyServer(function(input, output) {
    predictions <- reactive({
        tryCatch({
            pred <- NULL
            if(input$algo == "katz") {
                pred <- predictNextWord(input$words, as.numeric(input$resultNb))[, c("word", "prob")]
            } else {
                pred <- stupidPredictNextWord(input$words, as.numeric(input$resultNb))[, c("word", "prob")]
            }
            print(names(pred))
            pred$prob <- paste0(round(100 * pred$prob, 2), "%")
            names(pred) <- c("Word", "Probability")
            
            pred
        }, error = function(e) {
            e
        })
    })
    
    
    output$errorMsg <- renderText({
        if("simpleError" %in% class(predictions())) {
            predictions()$message
        } else { 
            ""
        }
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
    
    
    
    # output$map <- renderLeaflet({
    #     
    #     dept <- input$dept
    #     recip <- input$recip
    #     palette <- input$palette
    #     
    #     displayDept(dept, recip, palette)
    # })
})