
# packages ------------------------------------------------------------------------------------

#library(tidyverse)
#library(tidytext)
library(shiny)
library(shinyBS)
library(shinydashboard)
library(shinythemes)
library(shinyWidgets)
library(shinycssloaders)
library(Cairo)
#library(RColorBrewer)
#library(pheatmap)
#library(ggwordcloud)
#library(highcharter)
#library(htmlTable)
#library(feather)
library(MASS)
library(ggplot2)

# OPtions -----------

options(shiny.usecairo = TRUE)

# UI ---------------


ui <- dashboardPage(
  
  skin = "yellow",
  
  #### header ####
  
  dashboardHeader(
    
    title = "Prodetect"
    
  ),
  
  #### sidebar ####
  
  dashboardSidebar(
    
    sidebarMenu(id = "sidebar",
                
                menuItem("Analysis", tabName = "dashboard", icon = icon("chart-line", lib = "font-awesome")),
                menuItem("About Us", tabName = "aboutus", icon = icon("users", lib = "font-awesome"))
                
    ),
    
    hr(),
    
    conditionalPanel( condition = "input.sidebar == 'dashboard'"
      
    )
    
  ),
  
  #### body ####
  
  dashboardBody(
    
    #### css layout ####
    
    tags$head(tags$style(HTML('
                              /* logo */
                              .skin-blue .main-header .logo {
                              background-color: #606060;
                              }
                              
                              /* logo when hovered */
                              .skin-blue .main-header .logo:hover {
                              background-color: #606060;
                              }
                              
                              /* navbar (rest of the header) */
                              .skin-blue .main-header .navbar {
                              background-color: #606060;
                              }
                              
                              /* main sidebar */
                              .skin-blue .main-sidebar {
                              background-color: #ffffff;
                              }
                              
                              /* active selected tab in the sidebarmenu */
                              .skin-blue .main-sidebar .sidebar .sidebar-menu .active a{
                              background-color: #5b92e5;
                              color: #ffffff;
                              }
                              
                              /* other links in the sidebarmenu */
                              .skin-blue .main-sidebar .sidebar .sidebar-menu a{
                              background-color: #ffffff;
                              color: #000000;
                              }
                              
                              /* other links in the sidebarmenu when hovered */
                              .skin-blue .main-sidebar .sidebar .sidebar-menu a:hover{
                              background-color: #0A366B;
                              color: #ffffff;
                              }
                              /* toggle button when hovered  */
                              .skin-blue .main-header .navbar .sidebar-toggle:hover{
                              background-color: #0A366B;
                              }
                              .box.box-solid.box-primary>.box-header {
                              color:#fff;
                              }
                              
                              .box.box-solid.box-primary>.box-header {
                              color:#fff;
                              background:#606060
                              }
                              
                              .box.box-solid.box-primary{
                              border-bottom-color:#606060;
                              border-left-color:#606060;
                              border-right-color:#606060;
                              border-top-color:#606060;
                              }
                              ')
    )
    ),
    
    tabItems(
      
      #### content DB 1 ####
      
      tabItem(tabName = "dashboard",
              
              fluidRow(
                
                box(title = "Distribution of errors",
                    status = "primary",
                    width = 12,
                    height = "520px",
                    solidHeader = TRUE,
                    
                    plotOutput("errorplt2d"))
              

                )
              ),
    
      
      #### about us ####
      
      
      tabItem(tabName = "aboutus",
              
              includeMarkdown("AboutUs.Rmd")
              
      )
              
      )
    )
    )

# server --------------------------------------------------------------------------------------

server <- function(input, output) {
  
  obs <- rnorm(1000, mean = 0, sd = 4)
  mxcov <- cbind(c(17,2), c(2, 10))
  
  mvnobs <- mvrnorm(n = 100000, mu = c(0,0), Sigma = mxcov)
  
  df.mvnobs <- as.data.frame(mvnobs)
  colnames(df.mvnobs) <- c("a", "b")
  
  
  output$errorplot <- renderPlot(
    #obs <- rnorm(1000, mean = 0, d = 4),
    hist(obs)
  )
  
  output$errorplt2d <- renderPlot(
    ggplot(df.mvnobs, aes(x = a, y = b)) + geom_density_2d() + geom_point()
  )
  
}



# app ------------


shinyApp(ui = ui, server = server)
