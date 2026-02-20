module CNN_Accelerator_Engine (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [4:0]  input_image_index, 
    output reg  [3:0]  predicted_class,
    output reg         done
);

    typedef enum logic [2:0] {
        IDLE,
        START_DENSE,
        WAIT_DENSE,
        READ_OUTPUTS,
        START_ARGMAX,
        WAIT_ARGMAX,
        FINISH
    } state_t;

    state_t state;

    reg  dense_start;
    wire dense_done;
    reg  [31:0] read_addr;
    wire [31:0] read_data;

    reg signed [31:0] output_buffer [0:8];
    reg [3:0] index;

    reg argmax_start;
    wire argmax_done;
    wire [3:0] max_index;

    reg [31:0] wait_counter;

    reg [4:0] image [0:50];  
    reg [4:0] img;
    integer z, j;

    dense_layer_2x_128_to_9 layer_1_to_34_output (
        .clk(clk),
        .resetn(resetn),
        .start(dense_start),
        .input_image_index(input_image_index),
        .read_addr(index),
        .read_data(read_data),
        .done(dense_done)
    );

    argmax32_9_1 prediction_output_layer (
        .clk(clk),
        .resetn(resetn),
        .start(argmax_start),
        .data_in0(output_buffer[0]),
        .data_in1(output_buffer[1]),
        .data_in2(output_buffer[2]),
        .data_in3(output_buffer[3]),
        .data_in4(output_buffer[4]),
        .data_in5(output_buffer[5]),
        .data_in6(output_buffer[6]),
        .data_in7(output_buffer[7]),
        .data_in8(output_buffer[8]),
        .img(input_image_index),
        .max_index(max_index),
        .done(argmax_done));

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            state <= IDLE;
            dense_start <= 0;
            argmax_start <= 0;
            read_addr <= 0;
            predicted_class <= 0;
            done <= 0;
            index <= 0;
            wait_counter <= 0;
            img <= 0;
            j <= 0;
            for (z=0; z<51; z=z+1) image[z] <= 0;
            for (z=0; z<9; z=z+1) output_buffer[z] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    image[input_image_index] <= input_image_index;
                    if (start) begin
                        img <= 1; 
                        j <= 0;
                        for (z=0; z<10; z=z+1) begin
                            if (image[z] == z) begin
                                img <= image[z];
                                j <= 1;
                            end
                        end
                        if (j == 0) begin
                            dense_start <= 1;
                            state <= START_DENSE;
                            wait_counter <= 0;
                        end else begin
                            dense_start <= 1;
                            state <= START_DENSE;
                            wait_counter <= 0;
                        end
                    end
                end

                START_DENSE: begin
                    dense_start <= 0;
                    state <= WAIT_DENSE;
                end

                WAIT_DENSE: begin
                  if (wait_counter < 2**(2))
                        wait_counter <= wait_counter + 1;

                    if (dense_done) begin
                        index <= 0;
                        read_addr <= 0;
                        state <= READ_OUTPUTS;
                    end else if (wait_counter >= 2 begin
                        for (z=0; z<9; z=z+1) output_buffer[z] <= 0;
                        state <= READ_OUTPUTS;
                    end
                end

                READ_OUTPUTS: begin
                    output_buffer[index] <= read_data;
                    index <= index + 1;
                    if (index == 8)
                        state <= START_ARGMAX;
                end

                START_ARGMAX: begin
                    argmax_start <= 1;
                    state <= WAIT_ARGMAX;
                end

                WAIT_ARGMAX: begin
                    argmax_start <= 0;
                    if (argmax_done) begin
                        predicted_class <= max_index;
                        done <= 1;
                        state <= FINISH;
                    end
                end

                FINISH: begin
                    state <= FINISH;
                end
            endcase
        end
    end

endmodule
