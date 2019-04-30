ui = shiny::fluidPage(
   
   # Application title
   shiny::titlePanel("AutoxgboostMC"),
   
   # Sidebar with a slider input for number of bins 
   shiny::sidebarLayout(
      shiny::sidebarPanel(
         shiny::actionButton("startButton", "Fit!"),
         shiny::sliderInput("iterations",
                     "Iterations:",
                     min = 1,
                     max = 50,
                     value = 5)
      ),
      
      # Show a plot of the progress
      shiny::mainPanel(
         shiny::plotOutput("progressPlot"),
         shiny::plotOutput("paretoFront")
      )
   )
)

# Define server logic required to draw a histogram
server = function(input, output) {
   shiny::observeEvent(input$startButton, {
             axgb$fit(input$iterations)
    })

   output$progressPlot = shiny::renderPlot({
      NULL
   })
   output$paretoFront = shiny::renderPlot({
      plotly::ggplotly(axgb$plot_pareto_front())
   })
}
