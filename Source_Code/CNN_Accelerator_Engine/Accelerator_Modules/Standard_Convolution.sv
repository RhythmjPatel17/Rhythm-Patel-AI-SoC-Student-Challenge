module std_conv2d_224_224_3_32batches_batchnorm_relu6_1x_32ch_27pes(
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [4:0]  input_image_index,
    input  wire [31:0] read_addr,
    output wire [3:0]  read_data,
    output reg         done
);

    parameter IMG_W     = 224;
    parameter IMG_H     = 224;
    parameter IM_W      = 32;
    parameter IM_H      = 32;
    parameter NUM_CH    = 3;
    parameter NUM_FILT  = 32;
    parameter KSIZE     = 3;

    localparam OUT_SIZE   = IM_W * IM_H;
    localparam WEIGHT_NUM = NUM_FILT * NUM_CH * KSIZE * KSIZE;

    typedef enum logic [2:0] {
        IDLE,
        LOAD_IMAGE,
        WAIT_ONE_CYCLE,
        CONV2D_COMPUTE,
        DONE_STATE
    } state_t;

    state_t state;

    wire signed [7:0] weight_val;
    wire signed [7:0] bias_val;
    wire signed [7:0] scale_val;
    wire signed [7:0] shift_val;

    reg  [15:0] img_addr;
    reg  [15:0] weight_addr;
    reg  [5:0]  filt_addr;

    wire [31:0] conv_out_dout;
    reg  [31:0] conv_out_addr_a, conv_out_din_a;
    reg         conv_out_en_a, conv_out_en_b;
    reg  [3:0]  conv_out_we_a;
    reg [31:0] read_addr_n;
    integer r;

    assign read_data = (read_addr[2:0] == 3'd0) ? conv_out_dout[31:28] :
                       (read_addr[2:0] == 3'd1) ? conv_out_dout[27:24] :
                       (read_addr[2:0] == 3'd2) ? conv_out_dout[23:20] :
                       (read_addr[2:0] == 3'd3) ? conv_out_dout[19:16] :
                       (read_addr[2:0] == 3'd4) ? conv_out_dout[15:12] :
                       (read_addr[2:0] == 3'd5) ? conv_out_dout[11:8]  :
                       (read_addr[2:0] == 3'd6) ? conv_out_dout[7:4]   :
                                                  conv_out_dout[3:0];

    input_output_bram_wrapper conv2d_1x_bram (
        .BRAM_PORTA_0_addr(conv_out_addr_a),
        .BRAM_PORTA_0_clk(clk),
        .BRAM_PORTA_0_din(conv_out_din_a),
        .BRAM_PORTA_0_dout(),
        .BRAM_PORTA_0_en(conv_out_en_a),
        .BRAM_PORTA_0_we(conv_out_we_a),
        .BRAM_PORTB_0_addr((read_addr >> 3) * 4),
        .BRAM_PORTB_0_clk(clk),
        .BRAM_PORTB_0_din(32'd0),
        .BRAM_PORTB_0_dout(conv_out_dout),
        .BRAM_PORTB_0_en(conv_out_en_b),
        .BRAM_PORTB_0_we(4'd0)
    );

    integer i, j, idx;
    reg signed [7:0] image_val;
    reg [9:0] row, col;
    reg [12:0] out_idx;
    reg [5:0] filter_idx;
    reg signed [31:0] acc;
    reg [31:0] packed_data;
    reg [1:0] read_delay_counter;
    integer img_idx, weight_idx;
    integer ch_idx;
    integer bn_scaled, relu6_out;

    reg  signed [7:0] a_in[0:26], b_in[0:26];
    wire signed [15:0] pe_result[0:26];

    genvar pe;
    generate
        for (pe = 0; pe < 27; pe = pe + 1) begin : PE_ARRAY
            processing_element pe_inst (
                .a(a_in[pe]),
                .b(b_in[pe]),
                .result(pe_result[pe])
            );
        end
    endgenerate

    wire signed [7:0] image_val_rom[0:19];

    image_rom_input_32x32_1   img_rom_1  (.addr(img_addr), .data(image_val_rom[0]));
    image_rom_input_32x32_2   img_rom_2  (.addr(img_addr), .data(image_val_rom[1]));
    image_rom_input_32x32_3   img_rom_3  (.addr(img_addr), .data(image_val_rom[2]));
    image_rom_input_32x32_4   img_rom_4  (.addr(img_addr), .data(image_val_rom[3]));
    image_rom_input_32x32_5   img_rom_5  (.addr(img_addr), .data(image_val_rom[4]));
    image_rom_input_32x32_6   img_rom_6  (.addr(img_addr), .data(image_val_rom[5]));
    image_rom_input_32x32_7   img_rom_7  (.addr(img_addr), .data(image_val_rom[6]));
    image_rom_input_32x32_8   img_rom_8  (.addr(img_addr), .data(image_val_rom[7]));
    image_rom_input_32x32_9   img_rom_9  (.addr(img_addr), .data(image_val_rom[8]));
    image_rom_input_32x32_10  img_rom_10 (.addr(img_addr), .data(image_val_rom[9]));
    image_rom_input_32x32_11  img_rom_11 (.addr(img_addr), .data(image_val_rom[10]));
    image_rom_input_32x32_12  img_rom_12 (.addr(img_addr), .data(image_val_rom[11]));
    image_rom_input_32x32_13  img_rom_13 (.addr(img_addr), .data(image_val_rom[12]));
    image_rom_input_32x32_14  img_rom_14 (.addr(img_addr), .data(image_val_rom[13]));
    image_rom_input_32x32_15  img_rom_15 (.addr(img_addr), .data(image_val_rom[14]));
    image_rom_input_32x32_16  img_rom_16 (.addr(img_addr), .data(image_val_rom[15]));
    image_rom_input_32x32_17  img_rom_17 (.addr(img_addr), .data(image_val_rom[16]));
    image_rom_input_32x32_18  img_rom_18 (.addr(img_addr), .data(image_val_rom[17]));
    image_rom_input_32x32_19  img_rom_19 (.addr(img_addr), .data(image_val_rom[18]));
    image_rom_input_32x32_20  img_rom_20 (.addr(img_addr), .data(image_val_rom[19]));

    always @(*) begin
        case (input_image_index)
            5'd0 : image_val = image_val_rom[0];
            5'd1 : image_val = image_val_rom[1];
            5'd2 : image_val = image_val_rom[2];
            5'd3 : image_val = image_val_rom[3];
            5'd4 : image_val = image_val_rom[4];
            5'd5 : image_val = image_val_rom[5];
            5'd6 : image_val = image_val_rom[6];
            5'd7 : image_val = image_val_rom[7];
            5'd8 : image_val = image_val_rom[8];
            5'd9 : image_val = image_val_rom[9];
            5'd10: image_val = image_val_rom[10];
            5'd11: image_val = image_val_rom[11];
            5'd12: image_val = image_val_rom[12];
            5'd13: image_val = image_val_rom[13];
            5'd14: image_val = image_val_rom[14];
            5'd15: image_val = image_val_rom[15];
            5'd16: image_val = image_val_rom[16];
            5'd17: image_val = image_val_rom[17];
            5'd18: image_val = image_val_rom[18];
            5'd19: image_val = image_val_rom[19];
            default: image_val = 8'd0;
        endcase
    end

    weights_rom_1 w_rom    (.addr(weight_addr), .data(weight_val));
    biases_rom_1 bias_rom(.addr(filt_addr),   .data(bias_val));
    scale_rom_1   s_rom    (.addr(filt_addr),   .data(scale_val));
    shift_rom_1   sh_rom   (.addr(filt_addr),   .data(shift_val));

    always @(posedge clk) begin
        if (!resetn) begin
            state <= IDLE;
            done <= 0;
            conv_out_en_a <= 0;
            conv_out_en_b <= 0;
            conv_out_we_a <= 0;
            row <= 0; col <= 0; out_idx <= 0;
            filter_idx <= 0; acc <= 0; packed_data <= 0;
            read_delay_counter <= 0;
        end else begin
            conv_out_en_a <= 0;
            conv_out_we_a <= 0;
            done <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        row <= 0; col <= 0; out_idx <= 0;
                        filter_idx <= 0; acc <= 0; packed_data <= 0;
                        state <= LOAD_IMAGE;
                    end
                end

                LOAD_IMAGE: state <= WAIT_ONE_CYCLE;
                WAIT_ONE_CYCLE: state <= CONV2D_COMPUTE;

                CONV2D_COMPUTE: begin
                    if (row < IM_H) begin
                        if (col < IM_W) begin
                            idx = 0;
                            for (ch_idx = 0; ch_idx < NUM_CH; ch_idx = ch_idx + 1) begin
                                for (i = 0; i < 3; i = i + 1) begin
                                    for (j = 0; j < 3; j = j + 1) begin
                                        img_idx    = ch_idx * 1024 + (row + i) * IM_W + (col + j);
                                        weight_idx = filter_idx * 27 + ch_idx * 9 + i * 3 + j;
                                        img_addr   <= img_idx;
                                        weight_addr<= weight_idx;
                                        filt_addr  <= filter_idx;

                                        a_in[idx] <= (col + j < IM_W && row + i < IM_H) ? image_val : 8'd0;
                                        b_in[idx] <= weight_val;
                                        idx = idx + 1;
                                    end
                                end
                            end

                            acc = 0;
                            for (i = 0; i < 27; i = i + 1)
                                acc = acc + pe_result[i];
                            if (^acc === 1'bx) acc = 0;
                            acc = acc + bias_val;
                            acc = acc >>> 7;
                            acc = (acc > 127) ? 127 : ((acc < -128) ? -128 : acc);

                            bn_scaled = ((acc * scale_val) + shift_val) >>> 7;
                            relu6_out = (bn_scaled + 128) >>> 5;
                            relu6_out = (relu6_out > 6) ? 6 : (relu6_out < 0) ? 0 : relu6_out;

                            case (out_idx % 8)
                                0:  packed_data[31:28] = relu6_out[3:0];
                                1:  packed_data[27:24] = relu6_out[3:0];
                                2:  packed_data[23:20] = relu6_out[3:0];
                                3:  packed_data[19:16] = relu6_out[3:0];
                                4:  packed_data[15:12] = relu6_out[3:0];
                                5:  packed_data[11:8]  = relu6_out[3:0];
                                6:  packed_data[7:4]   = relu6_out[3:0];
                                7: begin
                                    packed_data[3:0] = relu6_out[3:0];
                                    conv_out_din_a   = packed_data;
                                    conv_out_addr_a  = ((filter_idx * 1024 + out_idx) >> 3) * 4;
                                    conv_out_en_a    = 1;
                                    conv_out_we_a    = 4'b1111;
                                    packed_data      = 0;
                                end
                            endcase

                            if ((out_idx == 1023) && (out_idx % 8 != 7)) begin
                                conv_out_din_a  <= packed_data;
                                conv_out_addr_a <= ((filter_idx * 1024 + out_idx) >> 3)*4;
                                conv_out_en_a   <= 1;
                                conv_out_we_a   <= 4'b1111;
                            end

                            out_idx <= out_idx + 1;
                            col <= col + 1;
                        end else begin
                            col <= 0;
                            row <= row + 1;
                        end
                    end else begin
                        if (filter_idx < NUM_FILT - 1) begin
                            filter_idx <= filter_idx + 1;
                            row <= 0; col <= 0; out_idx <= 0;
                            acc <= 0; packed_data <= 0;
                        end else begin
                            read_delay_counter <= 3;
                            state <= DONE_STATE;
                        end
                    end
                end

                DONE_STATE: begin
                    if (read_delay_counter > 0)
                        read_delay_counter <= read_delay_counter - 1;
                    else begin
                        conv_out_en_b <= 1;
                        done <= 1;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
