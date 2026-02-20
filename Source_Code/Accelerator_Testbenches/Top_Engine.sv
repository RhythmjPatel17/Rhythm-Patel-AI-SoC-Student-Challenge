module CNN_Accelerator_Engine_tb;

    logic clk;
    logic resetn;
    logic start;
    logic [4:0]  input_image_index;
    reg [3:0] predicted_class;
    logic done;

    initial clk = 0;
    always #5 clk = ~clk;  
    
    CNN_Accelerator_Engine uut (
        .clk(clk),
        .resetn(resetn),
        .start(start),
        .input_image_index(input_image_index),
        .predicted_class(predicted_class),
        .done(done)
    );

    
    string classes [0:8] = '{"CENTER","DONUT","EDGE-LOC","EDGE-RING","LOC","NEAR-FULL","RANDOM","SCRATCH","NONE"};

    initial begin
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 1;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 2;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 3;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 4;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 5;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 6;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 7;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 8;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 9;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 10;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 11;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 12;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 13;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 14;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 15;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 16;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 17;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 18;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 19;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        resetn = 0;
        start = 0;

        #10;
        resetn = 1;
        input_image_index = 20;
        
        #10;
        start = 1;
        #10;
        start = 0;

        #20;

        wait (done == 1);

        $display("Predicted Class Index: %0d", predicted_class);
        $display("Predicted Class Name : %s", classes[predicted_class]);

        #10;
        
        $finish;
    end
endmodule
