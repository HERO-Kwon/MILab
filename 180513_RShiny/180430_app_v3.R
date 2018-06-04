###########################################################
### Made By     :   HERO Kwon & JeeHoon Han             ###
### Date        :   2018.04.30                          ###
### Discription :   Application for Cell Voltage Design ###
### Version     :   0.8                                 ###
###########################################################

library(shiny)
library(ggplot2)
library(dplyr)
library(zoo)
library(reshape2)

# increase file upload limit to 1000MB
options(shiny.maxRequestSize = 1000*1024^2)

# unit constant(decimal point)
digit_volt <- 4
digit_capa <- 3

# Declare list variable
files_list <- list()
files_unit <- list()
files_info <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("mat","name","type","mode"))

# Algorithm Functions

# Setting Voltage, Capacity Unit
SetUnit <- function(list_item,k_digit,mode1,mode2){

    new_list_item <- list()
    
    volt_min <- min(sapply(list_item,function(x) min(x$volt)))
    volt_max <- max(sapply(list_item,function(x) max(x$volt)))
    capa_min <- min(sapply(list_item,function(x) min(x$capa)))
    capa_max <- max(sapply(list_item,function(x) max(x$capa)))

    for (df in list_item){
        if (mode1 == "volt"){
            df_volt <- round(df$volt,digit = k_digit)
            volt_min <- round(volt_min, digit = k_digit)
            volt_max <- round(volt_max, digit = k_digit)

            volt_unit <- round(data.frame(volt = seq(volt_min,volt_max,by=10^-k_digit)), digit=k_digit)

            df <- df %>% group_by(volt) %>% summarize(capa = mean(capa))
            df_new <- merge(volt_unit, df, by="volt", all=TRUE)
            #if (mode2=="cat") df_new$capa[1] <- capa_min
            #else if (mode2 == "ano") df_new$capa[1] <- capa_max
        }
        else if (mode1 == "capa"){
            df$capa <- round(df$capa,digit=k_digit)
            capa_min <- round(capa_min,digit=k_digit)
            capa_max <- round(capa_max,digit=k_digit)

            capa_unit <- round(data.frame(capa=seq(capa_min,capa_max,by=10^-k_digit)),digit=k_digit)

            df_new <- merge(capa_unit,df,by="capa",all=TRUE)
            if (mode2=="cat") df_new$volt[1] <- volt_min
            else if (mode2=="ano") df_new$volt[1] <- volt_max
        }

        df_new <- data.frame(na.approx(df_new))
        new_list_item <- c(new_list_item,list(df_new))
    }
    
    return(new_list_item)
}

# Blending active material
BlendActive <- function(item_unit,ratio){
    active_blend <- data.frame(volt=item_unit[[1]]$volt)
    active_blend$capa <- 0
    active_blend$capa_blend <- 0

    for (i in 1:length(ratio)){
        active_blend <- merge(active_blend,item_unit[[i]],by="volt",all=TRUE,suffixes=c("",as.character(i)))
        active_blend$capa_blend <- active_blend$capa_blend + active_blend[,i+3]*ratio[i]*0.01
    }
    active_blend <- active_blend[!is.na(active_blend$capa_blend),]
    active_blend <- active_blend[,-which(names(active_blend)=="capa")]
    active_blend$capa <- active_blend$capa_blend
    
    return(active_blend)
}

### Shiny UI ###

ui <- pageWithSidebar(

    # Always on panels
    headerPanel("Cell Voltage Design"),

    sidebarPanel(
        radioButtons("process",label=h4("Design Process"),
                    choices=list("1. File Input(*.csv)"=1,
                                 "2. Blending"=2,
                                 "3. N/P Ratio"=3,
                                 "4. Full Cell Voltage"=4,
                                 "5. Current Voltage Capacity"=5),
                                 selected=1),
        hr(),

        radioButtons("mat",label=h4("Choose Material"),
                    choices=list("Cathode"="Cathode","Anode"="Anode","OCV"="OCV"),
                    selected="Cathode"),
        fluidRow(verbatimTextOutput("value",placeholder=TRUE)),
        fluidRow(verbatimTextOutput("value1")),
        hr(),

        # Conditional Panels
        conditionalPanel(condition="input.process==1",
        
            fileInput("file",label=h4("Input CSV File"),
                    multiple=FALSE,
                    accept=c("text/csv",
                            "text/comma-separated-values,text/plain",
                            ".csv")),
            numericInput("skip_row",label=h4("Skip Row"),value=0),
            textInput("mat_name",label=h4("Material Name"),value=""),
            
            hr(),
            h4("Charge Data"),
            fluidRow(
                column(6,numericInput("capa_col_char",
                                    label=h5("Capacity Column"),
                                    value=1)),
                column(6,numericInput("volt_col_char",
                                    label=h5("Voltage Column"),
                                    value=2))
            ),
            hr(),
            
            h4("Discharge Data"),
            fluidRow(
                column(6,numericInput("capa_col_disc",
                                    label=h5("Capacity Column"),
                                    value=1)),
                column(6,numericInput("volt_col_disc",
                                    label=h5("Voltage Column"),
                                    value=2))
            ),
            hr(),
            actionButton("go_upload",label="Data Upload")
            ),
        
        conditionalPanel(condition="input.process==2",
            fluidRow(
                column(6,verbatimTextOutput("blend1")),
                column(6,verbatimTextOutput("blend2")),
                column(6,verbatimTextOutput("blend3")),
                column(6,verbatimTextOutput("blend4")),
                column(6,verbatimTextOutput("blend5"))
            ),
            fluidRow(
                column(4,numericInput("ratio1",label="1st",value=0)),
                column(4,numericInput("ratio2",label="2nd",value=0)),
                column(4,numericInput("ratio3",label="3rd",value=0)),
                column(4,numericInput("ratio4",label="4th",value=0)),
                column(4,numericInput("ratio5",label="5th",value=0))
            ),
            hr(),
            actionButton("go_ratio",label="Confirm Ratio")
        ),
        conditionalPanel(condition="input.process==3",
            numericInput("selected_cat",label=h4("Select Cathode"),value=""),
            numericInput("selected_ano",label=h4("Select Anode"),value=""),
            numericInput("selected_np",label=h4("N/P Ratio"),value=""),
            hr(),
            actionButton("go_np",label="Select Material & N/P")
        ),
        conditionalPanel(condition="input.process==4",
            fluidRow(actionButton("go_drawplot",label="Draw Voltage Plot")),
            br(),
            fluidRow(actionButton("go_fullcell",label="Draw FullCell Voltage"))
            ),
        conditionalPanel(condition="input.process==5",
            fileInput("file_1",label=h4("Input CSV File"),
                    multiple=FALSE,
                    accept=c("text/csv",
                            "text/comma-separated-values,text/plain",
                            ".csv")),
            numericInput("skip_row",label=h4("Skip Row"),value=0),
            textInput("mat_name_1",label=h4("OCV Material"),value=""),
            hr(),
            h4("Charge Data"),
            fluidRow(
                column(6,numericInput("capa_col_char_1",
                                    label=h5("Capacity Column"),
                                    value=1)),
                column(6,numericInput("volt_col_disc_1",
                                    label=h5("Voltage Column"),
                                    value=2))
            ),
            hr(),
            actionButton("capa",label="Data Upload")
        )
    ), # sidepanel

mainPanel(
    tabsetPanel(id="theTabs",
        tabPanel("Data",
                fluidRow(#Button
                    dataTableOutput("contents1"),
                    downloadButton("downloadData1","Download"),
                    br(),
                    dataTableOutput("contents2"),
                    downloadButton("downloadData2","Download")
                )),
        tabPanel("Graph",
                fluidRow(
                    plotOutput("cv_graph",
                        brush=brushOpts(id="plot_brush",fill="#ccc",direction="x"))),
                    plotOutput("fullcell_graph")
                )
        )
))

### Shiny Server

server <- function(input,output,session){
    ## Main Process
    observeEvent(input$process,{
        files_orig <- unique(files_info[files_info$type=="Orig",1:3])
        output$blend1 <- renderPrint({files_orig[files_orig$mat==input$mat,"name"][1]})
        output$blend2 <- renderPrint({files_orig[files_orig$mat==input$mat,"name"][2]})
        output$blend3 <- renderPrint({files_orig[files_orig$mat==input$mat,"name"][3]})
        output$blend4 <- renderPrint({files_orig[files_orig$mat==input$mat,"name"][4]})
        output$blend5 <- renderPrint({files_orig[files_orig$mat==input$mat,"name"][5]})
    })

    ## For Reading CSV ##
    # Read Data
    pass_data <- reactive({
        req(input$file)
        df <- read.csv(input$file$datapath,skip=input$skip_row,fileEncoding="EUC-KR")
        return(df)
    })
    observeEvent(input$file,{
        output$contents2 <- renderDataTable({})
        output$contents1 <- renderDataTable({
            table_data <- pass_data()
            names(table_data) <- paste(1:length(table_data),names(table_data),sep=")  ")
            return(table_data)
        })
    })
    pass_data_1 <- reactive({
        req(input$file_1)
        df_1 <- read.csv(input$file_1$datapath,skip=input$skip_row,fileEncoding="EUC-KR")
        return(df_1)
    })
    observeEvent(input$file_1,{
        output$contents2 <- renderDataTable({})
        output$contents1 <- renderDataTable({
            table_data <- pass_data_1()
            names(table_data) <- paste(1:length(table_data),names(table_data),sep=")  ")
            return(table_data)
        })
    })

    # Downloadable csv of selected dataset
    output$downloadData1 <- downloadHandler(
        filename=function(){
            "table_data1.csv"
        },
        content=function(file){
            write.csv(table_data1,file)
        }
    )
    output$downloadData2 <- downloadHandler(
        filename=function(){
            "table_data2.csv"
        },
        content=function(file){
            write.csv(table_data2,file)
        }
    )
    # Draw Graph
    output$cv_graph <- renderPlot({
        graph_data <- pass_data()
        graph_x <- graph_data[,input$capa_col]
        graph_y <- graph_data[,input$volt_col]

        ggolt(graph_data,aes(x=graph_x,y=graph_y))+geom_point()
    })

    # Proecess 1. Upload
    observeEvent(input$go_upload,{
        curr_data_char <- pass_data()[,c(input$capa_col_char,input$volt_col_char)]
        curr_data_disc <- pass_data()[,c(input$capa_col_disc,input$volt_col_disc)]

        curr_row <- 2*(input$go_upload[[1]]+input$go_ratio[[1]])-1

        names(curr_data_char) <- c("capa","volt")
        curr_data_char <- curr_data_char[complete.cases(curr_data_char),]
        files_list <<- append(files_list,list(curr_data_char))
        files_unit <<- append(files_unit,SetUnit(list(curr_data_char),digit_volt,"volt","cat"))

        files_info[curr_row,"mat"] <<- input$mat
        files_info[curr_row,"name"] <<- input$mat_name
        files_info[curr_row,"type"] <<- "Orig"
        files_info[curr_row,"mode"] <<- "Char"
        files_info <<- files_info[complete.cases(files_info),]

        names(curr_data_disc) <- c("capa","volt")
        curr_data_disc <- curr_data_disc[complete.cases(curr_data_disc),]
        files_list <<- append(files_list,list(curr_data_disc))
        files_unit <<- append(files_unit,SetUnit(list(curr_data_disc),digit_volt,"volt","cat"))

        files_info[curr_row+1,"mat"] <<- input$mat
        files_info[curr_row+1,"name"] <<- input$mat_name
        files_info[curr_row+1,"type"] <<- "Orig"
        files_info[curr_row+1,"mode"] <<- "Disc"
        
        # Output
        table_data1 <<- curr_data_char
        table_data2 <<- curr_data_disc

        output$value <- renderPrint({print(unique(files_info[,1:3]))})
        output$contents1 <- renderDataTable({table_data1},options=list(pageLength=10))
        output$contents2 <- renderDataTable({table_data2},options=list(pageLength=10))
        output$cv_graph <- renderPlot({
            ggplot(data=subset(curr_data_char,!is.na(capa)),aes(x=capa,y=volt)) +
            geom_point(color="salmon") +
            geom_point(data=subset(curr_data_disc,!is.na(capa)),aes(x=capa,y=volt),color="deep sky Blue")
        })
    })

    # Process2. Blending
    observeEvent(input$go_ratio,{
        blending_ratio <- c(input$ratio1,input$ratio2,input$ratio3,input$ratio4,input$ratio5)

        table_data <- data.frame()

        for(c_d in unique(files_info$mode)){
            files_ratio <- files_info[files_info$mat==input$mat,]
            mat_selected <- (files_info$type=="Orig")&(files_info$mode==c_d)&(files_info$mat==input$mat)

            files_selected <- files_unit[mat_selected]
            name_selected <- files_info[mat_selected,"name"]


            ratio_selected <- blending_ratio[1:length(mat_selected[mat_selected==TRUE])]
            ratio_selected <- ratio_selected*(100/sum(ratio_selected))

            # Blending
            name_blended <- paste(c(paste(name_selected,collapse=":"),paste(ratio_selected,collapse=":")),collapse=" = ")
            data_blended <- BlendActive(files_selected,ratio_selected)
            data_blended$mode <- c_d
            
            files_unit <<- append(files_unit,list(data_blended[,c("volt","capa")]))
            files_info <<- rbind(files_info,c(input$mat,name_blended,"Blnd",c_d))
            table_data <- rbind(table_data,data_blended)
        }

        # Output
        output$value <- renderPrint({print(unique(files_info[,1:3]))})
        table_data_melt <- melt(table_data,id.vars=c("mode","volt"))

        table_data1 <<- table_data %>% filter(mode=="Char")
        table_data2 <<- table_data %>% filter(mode=="Disc")

        output$contents1 <- renderDataTable({table_data1},options=list(pageLength=10))
        output$contents2 <- renderDataTable({table_data2},options=list(pageLength=10))
        output$cv_graph <- renderPlot({
            ggplot(data=table_data_melt %>% filter(mode=="Char"),aes(x=value,y=volt,group=variable,color=variable)) +
                geom_point() +
                geom_point(data=table_data_melt %>% filter(mode=="Disc"))
        })
    })

    # Process3. NP Ratios
    observeEvent(input$go_np,{
        volt_ano_char <- files_unit[[input$selected_ano]]['volt']
        capa_ano_char <- files_unit[[input$selected_ano]]['capa']
        volt_cat_char <- files_unit[[input$selected_cat]]['volt']
        capa_cat_char <- files_unit[[input$selected_cat]]['capa']

        volt_ano_disc <- files_unit[[input$selected_ano+1]]['volt']
        capa_ano_disc <- files_unit[[input$selected_ano+1]]['capa']
        volt_cat_disc <- files_unit[[input$selected_cat+1]]['volt']
        capa_cat_disc <- files_unit[[input$selected_cat+1]]['capa']

        np_data=max(capa_ano_disc)/max(capa_cat_disc)

        capa_np_char <- (capa_ano_char/np_data)*input$selected_np
        capa_np_disc <- (capa_ano_disc/np_data)*input$selected_np

        table_data1 <<- data.frame("volt"=volt_ano_disc,"capa_ano_raw"=capa_ano_disc,"capa_ano_np"=capa_np_disc$capa)

        # melting for ggplot grouping
        table_data_melt <- melt(table_data1,id.vars="volt")
        table_data_cat <- data.frame("volt"=volt_cat_disc,"value"=capa_cat_disc$capa)
        table_data_cat$variable <- "Cathode"

        data_ano_char <<- data.frame("volt"=volt_ano_char,"capa"=capa_np_char)
        data_cat_char <<- data.frame("volt"=volt_cat_char,"capa"=capa_cat_char)
        data_ano_disc <<- data.frame("volt"=volt_ano_disc,"capa"=capa_np_disc)
        data_cat_disc <<- data.frame("volt"=volt_cat_disc,"capa"=capa_cat_disc)

        # Output
        output$contents1 <- renderDataTable({table_data1},options=list(pageLength=10))
        output$contents2 <- renderDataTable({})
        output$cv_graph <- renderPlot({
            ggplot(data=table_data_melt,aes(x=value,y=volt,group=variable,color=variable))+
                geom_point(shape=3)+
                geom_point(data=table_data_cat,aes(x=value,y=volt))
        })
    })

    ## Process4. Voltage Setting
    observeEvent(input$go_drawplot,{
        cat_unit_capa <<- SetUnit(list(data_cat_char,data_cat_disc),digit_capa,"capa","cat")
        ano_unit_capa <<- SetUnit(list(data_ano_char,data_ano_disc),digit_capa,"capa","ano")

        voltage_plot <<- ggplot(data=ano_unit_capa[[1]],
                            aes(x=capa,y=volt))+
                            geom_point(color="salmon",shape=3)+
                            geom_point(data=cat_unit_capa[[1]],color="salmon")+
                            geom_point(data=ano_unit_capa[[2]],color="deep sky blue")+
                            geom_point(data=cat_unit_capa[[2]],color="deep sky blue")
        output$cv_graph <- renderPlot({
            voltage_plot
        })

        # Output
        output$value1 <- renderPrint({
            # Brushed Points
            brushed_cat_char <- brushedPoints(cat_unit_capa[[1]],input$plot_brush)
            brushed_ano_char <- brushedPoints(ano_unit_capa[[1]],input$plot_brush)
            brushed_cat_disc <- brushedPoints(cat_unit_capa[[2]],input$plot_brush)
            brushed_ano_disc <- brushedPoints(ano_unit_capa[[2]],input$plot_brush)
            brushed_pnt <<- list(brushed_cat_char,brushed_ano_char,brushed_cat_disc,brushed_ano_disc)

            char_left_cat <- brushed_cat_char[brushed_cat_char$capa==min(brushed_cat_char$capa),"volt"]
            char_right_cat <- brushed_cat_char[brushed_cat_char$capa==max(brushed_cat_char$capa),"volt"]
            char_left_ano <- brushed_ano_char[brushed_ano_char$capa==min(brushed_ano_char$capa),"volt"]
            char_right_ano <- brushed_ano_char[brushed_ano_char$capa==max(brushed_ano_char$capa),"volt"]
            disc_left_cat <- brushed_cat_disc[brushed_cat_disc$capa==min(brushed_cat_disc$capa),"volt"]
            disc_right_cat <- brushed_cat_disc[brushed_cat_disc$capa==max(brushed_cat_disc$capa),"volt"]
            disc_left_ano <- brushed_ano_disc[brushed_ano_disc$capa==min(brushed_ano_disc$capa),"volt"]
            disc_right_ano <- brushed_ano_disc[brushed_ano_disc$capa==max(brushed_ano_disc$capa),"volt"]

            char_left_fullcell <- char_left_cat - char_left_ano
            char_right_fullcell <- char_right_cat - char_right_ano
            disc_left_fullcell <- disc_left_cat - disc_left_ano
            disc_right_fullcell <- disc_right_cat - disc_right_ano

            print(paste("Voltage Range:",round(char_right_fullcell,3),"~",round(disc_left_fullcell,3)))
            print(paste("Capacity:",round((input$plot_brush$xmax-input$plot_brush$xmin),3)))
            print("Voltage@Charged")
            print(paste("Cathode:",round(char_right_cat,3)))
            print(paste("Anode:",round(char_right_ano,3)))
            print("Voltage@Discharged")
            print(paste("Cathode:",round(disc_left_cat,3)))
            print(paste("Anode:",round(disc_left_ano,3)))
        })
    })

    ## Process5. Current Voltage Capacity
    observeEvent(input$capa,{
        curr_data_char_1 <- pass_data_1()[,c(input$capa_col_char_1,input$volt_col_char_1)]
        curr_data_disc_1 <- pass_data_1()[,c(input$capa_col_disc_1,input$volt_col_disc_1)]

        curr_row_1 <- 2*(input$go_upload_1[[1]]+input$go_ratio_1[[1]])-1

        names(curr_data_char_1) <- c("capa","volt")
        curr_data_char_1 <- curr_data_char_1[complete.cases(curr_data_char_1),]
        files_list_1 <<- append(files_list_1,list(curr_data_char_1))
        files_unit_1 <<- append(files_unit_1,SetUnit(list(curr_data_char_1),digit_volt,"volt","cat"))

        files_info[curr_row,"mat"] <<- input$mat
        files_info[curr_row,"name"] <<- input$mat_name
        files_info[curr_row,"type"] <<- "Orig"
        files_info[curr_row,"mode"] <<- "Char"
        files_info <<- files_info[complete.cases(files_info),]

        names(curr_data_disc) <- c("capa","volt")
        curr_data_disc <- curr_data_disc[complete.cases(curr_data_disc),]
        files_list <<- append(files_list,list(curr_data_disc))
        files_unit <<- append(files_unit,SetUnit(list(curr_data_disc),digit_volt,"volt","cat"))

        files_info[curr_row+1,"mat"] <<- input$mat
        files_info[curr_row+1,"name"] <<- input$mat_name
        files_info[curr_row+1,"type"] <<- "Orig"
        files_info[curr_row+1,"mode"] <<- "Disc"
        
        # Output
        table_data1 <<- curr_data_char
        table_data2 <<- curr_data_disc

        output$value <- renderPrint({print(unique(files_info[,1:3]))})
        output$contents1 <- renderDataTable({table_data1},options=list(pageLength=10))
        output$contents2 <- renderDataTable({table_data2},options=list(pageLength=10))
        output$cv_graph <- renderPlot({
            ggplot(data=subset(curr_data_char,!is.na(capa)),aes(x=capa,y=volt))+
                geom_point(color="salmon")+
                geom_point(data=subset(curr_data_disc,!is.na(capa)),aes(x=capa,y=volt),color="deep sky Blue")
        })
    })

    ## Draw Full Cell Voltage
    observeEvent(input$go_fullcell,{
        fullcell_char <- brushed_pnt[[1]]
        fullcell_char$volt <- brushed_pnt[[1]]$volt - brushed_pnt[[2]]$volt
        fullcell_disc <- brushed_pnt[[3]]
        fullcell_disc$volt <- brushed_pnt[[3]]$volt - brushed_pnt[[4]]$volt

        table_data1 <<- fullcell_char
        table_data2 <<- fullcell_disc

        output$contents1 <- renderDataTable({fullcell_char},options=list(pageLength=10))
        output$contents2 <- renderDataTable({fullcell_disc},options=list(pageLength=10))

        output$fullcell_graph <- renderPlot({
            ggplot(data=fullcell_char,aes(x=capa,y=volt))+
                geom_point(color="salmon")+
                geom_point(data=fullcell_disc,color="deep sky Blue")
        })
    })
}
shinyApp(ui=ui,server=server)