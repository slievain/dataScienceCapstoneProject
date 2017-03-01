library(shiny)

shinyUI(fluidPage(
    theme = "bootstrap.min.css",
    
    tags$head(tags$style(type="text/css", "
        html, body { width:100%; height:100% }
        #loadmessage {
            position: fixed;
            top: 0px;
            left: 0px;
            width: 100%;
            padding: 5px 0px 5px 0px;
            text-align: center;
            font-weight: bold;
            font-size: 100%;
            color: #000000;
            background-color: #CCFF66;
            z-index: 105;
        }
        #errorMsg {
            color: red;
        }")),
    
    conditionalPanel(condition = "$('html').hasClass('shiny-busy')",
                     tags$div("Loading...", id = "loadmessage")),
    
    titlePanel("Coursera Data Science Specialisation - Capstone Project"), 
    
    sidebarLayout(
        sidebarPanel(
            textInput("words", "Phrase (2 words exactly): ", "one of"),
            
            radioButtons("algo", "Algorithm:",
                         c("Stupid Back-off Model" = "stupid",
                           "Katz's Back-off Model" = "katz")),
            
            div(style = "display: inline-block;vertical-align:-webkit-baseline-middle; width: 40px", "Show"),
            div(style = "display: inline-block;vertical-align:-webkit-baseline-middle; width: 76px", selectInput("resultNb", "", c("5", "10", "15", "20"), selected = "10", width = 70)),
            div(style = "display: inline-block;vertical-align:-webkit-baseline-middle;", "entries"),
            
            submitButton("Submit")
        ), 
        
        mainPanel(
            tabsetPanel(
                tabPanel("Predictions",
                         br(),
                         textOutput("errorMsg"),
                         dataTableOutput('prediction')
                ),
                tabPanel("User Guide",
                         br(),
                         p("Using this app is very simple! "),
                         tags$ul(
                             tags$li("Enter 2 words exactly in the text input"),
                             tags$li("Press the Submit button."))
                )
            )
        )
    )
))