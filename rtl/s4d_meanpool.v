`timescale 1ns / 1ps
`default_nettype none

module s4d_meanpool #(
    parameter integer H      = 128,
    parameter integer LOG2_H = 7,
    parameter integer LOG2_L = 10,        // sCIFAR (era 12/Speech Commands)
    parameter integer W_IN   = 16,
    parameter integer W_ACC  = 32,
    parameter integer W_OUT  = 16
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    in_valid,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last_ch,
    input  wire                    in_last_seq,

    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last,
    output wire                    busy
);

    reg signed [W_ACC-1:0] acc [0:H-1];

    reg [LOG2_H-1:0] ch;
    reg              first;
    reg              drain;
    reg [LOG2_H-1:0] ocnt;

    assign busy = drain;

    wire signed [W_ACC-1:0] in_ext = {{(W_ACC-W_IN){in_data[W_IN-1]}}, in_data};
    wire signed [W_ACC-1:0] acc_cur = acc[ch];

    always @(posedge clk)
        if (rst_n && in_valid) acc[ch] <= first ? in_ext : (acc_cur + in_ext);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch    <= {LOG2_H{1'b0}};
            first <= 1'b1;

        end else if (in_valid) begin

            if (in_last_ch) begin
                ch    <= {LOG2_H{1'b0}};
                first <= in_last_seq;
            end else begin
                ch <= ch + 1'b1;
            end
        end
    end

    wire signed [W_ACC-1:0] rnd     = {{(W_ACC-1){1'b0}}, 1'b1} <<< (LOG2_L-1);
    wire signed [W_ACC-1:0] avg     = (acc[ocnt] + rnd) >>> LOG2_L;

    localparam signed [W_ACC-1:0] SAT_MAX = (1 <<< (W_OUT-1)) - 1;
    localparam signed [W_ACC-1:0] SAT_MIN = -(1 <<< (W_OUT-1));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain     <= 1'b0;
            ocnt      <= {LOG2_H{1'b0}};
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_last  <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            if (!drain) begin
                if (in_valid && in_last_seq) begin
                    drain <= 1'b1;
                    ocnt  <= {LOG2_H{1'b0}};
                end
            end else begin
                out_valid <= 1'b1;
                out_last  <= (ocnt == H-1);

                if      (avg > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
                else if (avg < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
                else                    out_data <= avg[W_OUT-1:0];

                if (ocnt == H-1) drain <= 1'b0;
                else             ocnt  <= ocnt + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
