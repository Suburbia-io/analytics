for (package in c("shiny", "ggplot2", "dplyr", "scales")) {
    if (!library(package, character.only = TRUE, logical.return = TRUE)) {
        install.packages(package)
    }
}

library(shiny)
library(ggplot2)
library(dplyr)
library(scales)


# Define UI for application that draws a histogram
ui <- fluidPage(

    titlePanel("Stability settings selector"),

    sidebarLayout(
        sidebarPanel(
            fileInput("merchant_csv", "Choose merchant CSV file",
                      accept = c(
                          "text/csv",
                          "text/comma-separated-values,text/plain",
                          ".csv")
            ),

            checkboxGroupInput("vendors", "Vendors", inline = TRUE),

            sliderInput("lifespan",
                        "Minimum lifespan:",
                        min = 1,
                        max = 1200,
                        value = 365),
            sliderInput("activity",
                        "Minimum activity:",
                        min = 0,
                        max = 1,
                        value = 0.67),
            sliderInput("volatility",
                        "Maximum volatility:",
                        min = 0,
                        max = 1,
                        value = 0.5),
            sliderInput("max_gap",
                        "Maximum allowed gap (days):",
                        min = 0,
                        max = 365,
                        value = 21),

            width = 3,
        ),

        # Show a plot of the generated distribution
        mainPanel(
            tabsetPanel(
                tabPanel(
                    "Charts",
                    br(), br(),
                    strong(textOutput("merchants")),
                    textOutput("lines"),
                    br(), br(),
                    fluidRow(
                        column(4, plotOutput("lines_plot")),
                        column(4, plotOutput("activity_plot"), plotOutput("volatility_plot")),
                        column(4, plotOutput("lifespan_plot"), plotOutput("longest_gap_plot")),
                    ),
                    br(), br(),
                    downloadButton('download_data', 'Download results')
                ),
                tabPanel(
                    "Statistics by vendor",
                    br(), br(),
                    fluidRow(
                        tableOutput("table_final"),
                        tableOutput("table_by_vendor")
                    )
                )
            )
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {

    stats_table <- reactive({
        in_file <- input$merchant_csv

        if (is.null(in_file))
            return(NULL)

        read.csv(in_file$datapath)
    })

    observeEvent(stats_table(), {
        all_vendors <- unique(stats_table()$vendor)
        updateCheckboxGroupInput(session, "vendors", choices = all_vendors, selected = all_vendors)
    })

    # Selecting merchants ---------------------------------------------------
    stats_table_filtered <- reactive({
        req(stats_table())
        stats_table() %>%
            filter(vendor %in% input$vendors) %>%
            mutate(
                included = (
                lifespan >= input$lifespan &
                activity >= input$activity &
                volatility <= input$volatility &
                longest_gap <= input$max_gap
            )
        )
    })


    # Charts ----------------------------------------------------------------
    output$activity_plot <- renderPlot({
        req(stats_table_filtered())
        ggplot(stats_table_filtered(),
               aes(x = activity,
                   fill = included)) +
            geom_histogram() +
            scale_x_continuous(labels = percent, limits = c(0, 1)) +
            labs(title = "Distribution of merchants by activity",
                 x = "Activity",
                 y = "# of merchants",
                 fill = "Selected")
    })

    output$lifespan_plot <- renderPlot({
        req(stats_table_filtered())
        ggplot(stats_table_filtered(),
               aes(x = lifespan,
                   fill = included)) +
            geom_histogram(bins = 30) +
            xlim(0, 1200) +
            labs(title = "Distribution of merchants by lifespan",
                 x = "Lifespan (days)",
                 y = "# of merchants",
                 fill = "Selected")
    })

    output$longest_gap_plot <- renderPlot({
        req(stats_table_filtered())
        ggplot(stats_table_filtered(),
               aes(x = longest_gap,
                   fill = included)) +
            scale_x_continuous(limits = c(0, 365)) +
            geom_histogram(bins = 30) +
            labs(title = "Distribution of merchants by maximum gap",
                 x = "Maximum gap (days)",
                 y = "# of merchants",
                 fill = "Selected")
    })

    output$volatility_plot <- renderPlot({
        req(stats_table_filtered())
        ggplot(stats_table_filtered(),
               aes(x = volatility,
                   fill = included)) +
            geom_histogram(bins = 30) +
            scale_x_continuous(labels = percent, limits = c(0, 1)) +
            labs(title = "Distribution of merchants by relative volatility",
                 x = "Volatility",
                 y = "# of merchants",
                 fill = "Selected")
    })

    output$lines_plot <- renderPlot({
        req(stats_table_filtered())
        ggplot(stats_table_filtered(),
               aes(x = n_lines,
                   fill = included)) +
            geom_histogram(bins = 30) +
            scale_x_continuous(labels = comma, limits = c(0, 1500000)) +
            labs(title = "Distribution of merchants by # of lines",
                 x = "# of lines",
                 y = "# of merchants",
                 fill = "Selected")
    })


    # Computing statistics ----------------------------------------------------
    selection_stats_overall <- reactive({
        req(stats_table_filtered())
        stats_table_filtered() %>%
            group_by(included) %>%
            summarize(n_merchants = n(),
                      n_lines = sum(n_lines)) %>%
            ungroup() %>%
            mutate(
                n_merchants_ratio = percent(n_merchants / sum(n_merchants)),
                n_lines_ratio = percent(n_lines / sum(n_lines))
            )
    })
    selection_stats_by_vendor <- reactive({
        req(stats_table_filtered())
        stats_table_filtered() %>%
            group_by(included, vendor) %>%
            summarize(n_merchants = n(),
                      n_lines = sum(n_lines)) %>%
            ungroup() %>%
            mutate(
                n_merchants_ratio = percent(n_merchants / sum(n_merchants)),
                n_lines_ratio = percent(n_lines / sum(n_lines))
            )
    })

    output$table_by_vendor <- renderTable({
        req(selection_stats_by_vendor())
        selection_stats_by_vendor() %>%
            mutate(included = if_else(included, 'yes', 'no')) %>%
            select(vendor, included, n_merchants, n_merchants_ratio, n_lines, n_lines_ratio) %>%
            arrange(vendor, desc(included))
    })

    output$table_final <- renderTable({
        req(selection_stats_overall(), selection_stats_by_vendor())

        overall <- selection_stats_overall() %>%
            mutate(
                vendor = "[all]",
            ) %>%
            filter(included == TRUE)
        by_vendor <- selection_stats_by_vendor() %>%
            filter(included == TRUE)
        combined <- bind_rows(overall, by_vendor)
        combined %>%
            mutate(summary_merchants = paste0(comma(n_merchants), " (", n_merchants_ratio, ")"),
                   summary_lines = paste0(comma(n_lines), " (", n_lines_ratio, ")")
            ) %>%
            select(vendor, merchants = summary_merchants, lines = summary_lines) %>%
            transpose_df()
    })

    transpose_df <- function(df) {
        col_list <- unname(as.list(df))
        new_names <- col_list[1][[1]]
        new_rows <- col_list[-1]
        for (i in 1:length(new_rows)) {
            row_copy <- new_rows[[i]]
            names(row_copy) <- new_names
            new_rows[[i]] <- row_copy
        }
        bind_rows(!!!new_rows)
    }

    output$merchants <- renderText({
        req(selection_stats_overall())
        paste0(selection_stats_overall() %>% filter(included) %>% pull(n_merchants_ratio), " merchants")
    })

    output$lines <- renderText({
        req(selection_stats_overall())
        paste0(selection_stats_overall() %>% filter(included) %>% pull(n_lines_ratio), " lines")
    })


    # Download results ------------------------------------------------
    output$download_data <- downloadHandler(
        filename = function() {
            paste0('merchants-', Sys.Date(), '.zip')
        },

        content = function(fname) {
            req(selection_stats_by_vendor())
            tmpdir <- tempdir()

            # write selected merchants to tempfile ----
            to_save <- stats_table_filtered() %>%
                filter(included == TRUE) %>%
                select(vendor, merchant)
            write.csv(
                to_save,
                paste0(tmpdir, "/selected_merchants.csv"),
                quote = FALSE,
                row.names = FALSE
            )

            # write metadata to tempfile ---------------
            write_metadata <- function(con, input) {
                try(file.remove(con))
                add_to_file <- function(x) write(x, con, append = TRUE)
                add_to_file(paste("Generated:", Sys.time(), "\n"))
                add_to_file(paste("Minimum lifespan:", input$lifespan))
                add_to_file(paste("Minimum activity:", input$activity))
                add_to_file(paste("Maximum volatility:", input$volatility))
                add_to_file(paste("Maximum gap:", input$max_gap))
                add_to_file(paste("\nSelected vendors:", paste(input$vendors, collapse = ", ")))
            }
            write_metadata(paste0(tmpdir, "/metadata.txt"), input = input)

            # write statistics to tempfile ------------
            write.csv(
                selection_stats_by_vendor(),
                paste0(tmpdir, "/statistics.csv"),
                quote = FALSE,
                row.names = FALSE
            )

            # write sample of input data --------------
            write.csv(
                stats_table() %>% head(10),
                paste0(tmpdir, "/input_head.csv"),
                quote = FALSE,
                row.names = FALSE
            )

            # zip together ----------------------------
            fs <- c("selected_merchants.csv", "metadata.txt", "statistics.csv", "input_head.csv")
            zip(zipfile = fname, files = paste0(tmpdir, "/", fs), flags = "-j")
        },
        contentType = "application/zip"
    )

}

# Run the application
shinyApp(ui = ui, server = server, options = list(host = "0.0.0.0", port = 5555))
