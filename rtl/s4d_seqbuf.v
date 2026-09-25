`timescale 1ns / 1ps
`default_nettype none

module s4d_seqbuf #(
    parameter integer W      = 16,
    parameter integer NWORDS = 131072,
    parameter integer AW     = 17
)(
    input  wire                clk,
    input  wire                rst_n,

    input  wire                wr_en,
    input  wire [AW-1:0]       wr_addr,
    input  wire signed [W-1:0] wr_data,

    input  wire                rd_en,
    input  wire [AW-1:0]       rd_addr,
    output reg  signed [W-1:0] rd_data,

    input  wire                chk_en,
    output reg                 err_ovw
);

    (* ram_style = "block" *)
    reg [W-1:0] mem [0:NWORDS-1];

    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        if (rd_en) rd_data <= $signed(mem[rd_addr]);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                                    err_ovw <= 1'b0;
        else if (chk_en && wr_en && (wr_addr >= rd_addr)) err_ovw <= 1'b1;
    end

endmodule

`default_nettype wire
