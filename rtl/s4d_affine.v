`timescale 1ns / 1ps
`default_nettype none

module s4d_affine #(
    parameter integer H      = 128,
    parameter integer LOG2_H = 7,
    parameter integer W_IN   = 16,
    parameter integer W_OUT  = 16,
    parameter integer W_W    = 16,
    parameter integer W_B    = 32,
    parameter integer W_ACC  = 34,
    parameter integer SH     = 11,

    parameter integer LOADABLE = 0,
    parameter         W_FILE = "l6_bn_w.mem",  // modulo riusato per 6+1 ruoli diversi
    parameter         B_FILE = "l6_bn_b.mem"   // (nessun nome e' "il" nome giusto): punto
                                                // al BN finale del decoder, l'unico caso in
                                                // cui questo default viene davvero letto
                                                // (LOADABLE=0). Chi lo usa per un layer deve
                                                // sempre sovrascrivere esplicitamente.
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    ld_we_w,
    input  wire                    ld_we_b,
    input  wire [LOG2_H-1:0]       ld_addr,
    input  wire signed [W_W-1:0]   ld_w,
    input  wire signed [W_B-1:0]   ld_b,

    input  wire                    ld_we_sh,
    input  wire [5:0]              ld_sh,

    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last_seq,

    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last_ch,
    output reg                     out_last_seq
);

    reg signed [W_W-1:0] w_rom [0:H-1];
    reg signed [W_B-1:0] b_rom [0:H-1];

    generate
    if (LOADABLE == 0) begin : g_init
        initial begin
            $readmemh(W_FILE, w_rom);
            $readmemh(B_FILE, b_rom);
        end
    end else begin : g_load
        always @(posedge clk) begin
            if (ld_we_w) w_rom[ld_addr] <= ld_w;
            if (ld_we_b) b_rom[ld_addr] <= ld_b;
        end
    end
    endgenerate

    reg [5:0] sh_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                             sh_r <= SH[5:0];
        else if ((LOADABLE != 0) && ld_we_sh)   sh_r <= ld_sh;
    end

    wire en = out_ready;
    assign in_ready = out_ready;

    wire acc_in = in_valid & in_ready;

    reg [LOG2_H-1:0] ch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          ch <= {LOG2_H{1'b0}};
        else if (acc_in)     ch <= (ch == H-1) ? {LOG2_H{1'b0}} : ch + 1'b1;
    end

    reg signed [W_IN-1:0] x0;
    reg signed [W_W-1:0]  w0;
    reg signed [W_B-1:0]  b0;
    reg [LOG2_H-1:0]      ch0;
    reg                   v0, ls0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v0 <= 1'b0;
            ls0 <= 1'b0;
        end else if (en) begin
            v0  <= acc_in;
            ls0 <= acc_in & in_last_seq;
        end
    end

    always @(posedge clk) begin
        if (en) begin
            x0  <= in_data;
            w0  <= w_rom[ch];
            b0  <= b_rom[ch];
            ch0 <= ch;
        end
    end

    reg signed [W_ACC-1:0] acc1;
    reg [LOG2_H-1:0]       ch1;
    reg                    v1, ls1;

    wire signed [W_W+W_IN-1:0] prod = w0 * x0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            ls1 <= 1'b0;
        end else if (en) begin
            v1  <= v0;
            ls1 <= ls0;
        end
    end

    always @(posedge clk) begin
        if (en) begin
            acc1 <= $signed({{(W_ACC-W_W-W_IN){prod[W_W+W_IN-1]}}, prod})
                  + $signed({{(W_ACC-W_B){b0[W_B-1]}}, b0});
            ch1  <= ch0;
        end
    end

    wire signed [W_ACC-1:0] RND = (sh_r > 0)
                                          ? ({{(W_ACC-1){1'b0}}, 1'b1} <<< (sh_r-1))
                                          : {W_ACC{1'b0}};
    localparam signed [W_ACC-1:0] SAT_MAX = (1 <<< (W_OUT-1)) - 1;
    localparam signed [W_ACC-1:0] SAT_MIN = -(1 <<< (W_OUT-1));

    wire signed [W_ACC-1:0] shifted = (acc1 + RND) >>> sh_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_data     <= {W_OUT{1'b0}};
            out_last_ch  <= 1'b0;
            out_last_seq <= 1'b0;
        end else if (en) begin
            out_valid    <= v1;
            out_last_ch  <= v1 && (ch1 == H-1);
            out_last_seq <= v1 && (ch1 == H-1) && ls1;

            if      (shifted > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
            else if (shifted < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
            else                        out_data <= shifted[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
