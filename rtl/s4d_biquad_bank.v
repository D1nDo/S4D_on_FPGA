`timescale 1ns / 1ps
`default_nettype none

module s4d_biquad_bank #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,
    parameter integer LOG2_NCH = 4,

    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,
    parameter integer W_U      = 16,
    parameter integer F_U      = 11,
    parameter integer W_A      = 24,
    parameter integer F_A      = 22,
    parameter integer W_B      = 16,
    parameter integer F_B      = 15,
    parameter integer W_S      = 32,
    parameter integer F_Y      = 19,
    parameter integer W_D      = 16,
    parameter integer F_D      = 14,

    parameter integer W_BSH    = 5,
    parameter integer W_ACC    = 40,
    parameter integer W_OUT    = 16,
    parameter integer F_OUT    = 11
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    cfg_we,
    input  wire [LOG2_NU-1:0]      cfg_unit,
    input  wire [LOG2_NB-1:0]      cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]  cfg_data,

    input  wire                    cfgd_we,
    input  wire [LOG2_H-1:0]       cfgd_ch,
    input  wire signed [W_D-1:0]   cfgd_data,
    input  wire [W_BSH-1:0]        cfgd_bsh,

    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_U-1:0]   in_data,
    input  wire                    in_last_seq,

    output reg                     out_valid,
    input  wire                    out_ready,
    output reg  signed [W_OUT-1:0] out_data,
    output reg  [LOG2_H-1:0]       out_ch,
    output reg                     out_last,

    output wire                    ovf
);

    localparam integer SH_O = F_Y - F_OUT;

    localparam signed [W_ACC-1:0] OMAX = {{(W_ACC-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [W_ACC-1:0] OMIN = {{(W_ACC-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    reg [2*H*W_U-1:0] ubuf;
    reg               wsel, rsel;
    reg [LOG2_H-1:0]  wcnt;
    reg               buf_full;

    assign in_ready = !buf_full;

    reg                     start;

    reg                     clr;
    reg                     clr_next;
    reg                     buf_lastseq;
    wire [NUNITS-1:0]       u_busy;
    wire [NUNITS-1:0]       u_ovalid;
    wire [NUNITS*W_ACC-1:0] u_oacc;
    wire [NUNITS-1:0]       u_ovf;

    assign ovf = |u_ovf;

    genvar j;
    generate
        for (j = 0; j < NUNITS; j = j + 1) begin : g_unit

            wire [NCH*W_U-1:0] u_slice =
                ubuf[(rsel*H + j*NCH)*W_U +: NCH*W_U];

            s4d_biquad_unit #(
                .NCH(NCH), .MODES(MODES),
                .LOG2_NB(LOG2_NB), .LOG2_MOD(LOG2_MOD), .LOG2_CH(LOG2_NCH),
                .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A),
                .W_B(W_B), .F_B(F_B), .W_S(W_S), .F_Y(F_Y),
                .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH), .W_ACC(W_ACC)
            ) u_bq (
                .clk       (clk),
                .rst_n     (rst_n),
                .cfg_we    (cfg_we && (cfg_unit == j[LOG2_NU-1:0])),
                .cfg_addr  (cfg_addr),
                .cfg_data  (cfg_data),
                .cfgd_we   (cfgd_we && (cfgd_ch[LOG2_H-1:LOG2_NCH] == j[LOG2_NU-1:0])),
                .cfgd_addr (cfgd_ch[LOG2_NCH-1:0]),
                .cfgd_data (cfgd_data),
                .cfgd_bsh  (cfgd_bsh),
                .start     (start),
                .clr       (clr),
                .u_vec     (u_slice),
                .busy      (u_busy[j]),
                .out_valid (u_ovalid[j]),
                .out_acc   (u_oacc[j*W_ACC +: W_ACC]),
                .ovf       (u_ovf[j])
            );
        end
    endgenerate

    reg [H*W_ACC-1:0] res;
    reg [LOG2_NCH-1:0] slot_cnt;

    wire res_ev    = u_ovalid[0];
    wire pass_done = res_ev && (slot_cnt == NCH-1);

    integer jj;
    always @(posedge clk) begin
        if (res_ev)
            for (jj = 0; jj < NUNITS; jj = jj + 1)
                res[(jj*NCH + slot_cnt)*W_ACC +: W_ACC] <=
                    u_oacc[jj*W_ACC +: W_ACC];
    end

    reg pass_active;
    reg stream;
    reg [LOG2_H-1:0] ocnt;

    wire out_free = !out_valid || out_ready;
    wire beat     = stream && out_free;
    wire last_beat = beat && (ocnt == H-1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wsel        <= 1'b0;
            rsel        <= 1'b0;
            wcnt        <= {LOG2_H{1'b0}};
            buf_full    <= 1'b0;
            start       <= 1'b0;
            pass_active <= 1'b0;
            stream      <= 1'b0;
            ocnt        <= {LOG2_H{1'b0}};
            slot_cnt    <= {LOG2_NCH{1'b0}};
            clr         <= 1'b0;
            clr_next    <= 1'b1;
            buf_lastseq <= 1'b0;
        end else begin
            start <= 1'b0;
            clr   <= 1'b0;

            if (in_valid && in_ready) begin
                ubuf[(wsel*H + wcnt)*W_U +: W_U] <= in_data;
                if (wcnt == H-1) begin
                    wcnt        <= {LOG2_H{1'b0}};
                    buf_full    <= 1'b1;
                    wsel        <= ~wsel;

                    buf_lastseq <= in_last_seq;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
            end

            if (buf_full && !pass_active && !stream && !(|u_busy)) begin
                start       <= 1'b1;
                clr         <= clr_next;
                clr_next    <= buf_lastseq;
                rsel        <= ~wsel;
                buf_full    <= 1'b0;
                pass_active <= 1'b1;
                slot_cnt    <= {LOG2_NCH{1'b0}};
            end

            if (res_ev) slot_cnt <= slot_cnt + 1'b1;

            if (pass_done) begin
                pass_active <= 1'b0;
                stream      <= 1'b1;
                ocnt        <= {LOG2_H{1'b0}};
            end else if (last_beat) begin
                stream <= 1'b0;
                ocnt   <= {LOG2_H{1'b0}};
            end else if (beat) begin
                ocnt <= ocnt + 1'b1;
            end
        end
    end

    wire signed [W_ACC-1:0] acc_now = $signed(res[ocnt*W_ACC +: W_ACC]);

    wire signed [W_ACC-1:0] rnd  = {{(W_ACC-1){1'b0}}, 1'b1} <<< (SH_O-1);
    wire signed [W_ACC-1:0] yrnd = (acc_now + rnd) >>> SH_O;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_ch    <= {LOG2_H{1'b0}};
            out_last  <= 1'b0;
        end else if (out_free) begin
            out_valid <= beat;
            out_ch    <= ocnt;
            out_last  <= last_beat;

            if      (yrnd > OMAX) out_data <= OMAX[W_OUT-1:0];
            else if (yrnd < OMIN) out_data <= OMIN[W_OUT-1:0];
            else                  out_data <= yrnd[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
