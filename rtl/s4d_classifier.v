`timescale 1ns / 1ps
`default_nettype none

module s4d_classifier #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer K        = 10,      // sCIFAR: 10 classi (era 35/Speech Commands)
    parameter integer LOG2_K   = 4,       // ceil(log2(10)) (era 6/Speech Commands)
    parameter integer W_IN     = 16,
    parameter integer W_CIN    = 8,
    parameter integer W_CLS    = 8,
    parameter integer W_ACC    = 32,
    parameter integer RQ_SHIFT = 6,
    parameter         W_FILE   = "cls_w.mem",
    parameter         B_FILE   = "cls_b.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    in_valid,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last,

    output reg                     out_valid,
    output reg  [LOG2_K-1:0]       out_idx,
    output reg  signed [W_ACC-1:0] out_logit,
    output reg                     out_last,

    output reg                     class_valid,
    output reg  [LOG2_K-1:0]       class_idx
);

    reg [K*W_CLS-1:0]      w_rom [0:H-1];
    reg signed [W_ACC-1:0] b_rom [0:K-1];

    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    localparam signed [W_IN:0] CSAT_MAX = (1 <<< (W_CIN-1)) - 1;
    localparam signed [W_IN:0] CSAT_MIN = -(1 <<< (W_CIN-1));

    wire signed [W_IN:0] rq_rnd = (RQ_SHIFT == 0)
                                ? {(W_IN+1){1'b0}}
                                : ({{W_IN{1'b0}}, 1'b1} <<< (RQ_SHIFT-1));
    wire signed [W_IN:0] rq_ext = {in_data[W_IN-1], in_data};
    wire signed [W_IN:0] rq_val = (rq_ext + rq_rnd) >>> RQ_SHIFT;

    reg signed [W_CIN-1:0] xq;
    reg                    xq_valid, xq_last;
    reg [LOG2_H-1:0]       hcnt;
    reg [K*W_CLS-1:0]      w_row;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            xq       <= {W_CIN{1'b0}};
            xq_valid <= 1'b0;
            xq_last  <= 1'b0;
            hcnt     <= {LOG2_H{1'b0}};
            w_row    <= {(K*W_CLS){1'b0}};
        end else begin
            xq_valid <= in_valid;
            xq_last  <= in_valid && in_last;
            w_row    <= w_rom[hcnt];

            if      (rq_val > CSAT_MAX) xq <= CSAT_MAX[W_CIN-1:0];
            else if (rq_val < CSAT_MIN) xq <= CSAT_MIN[W_CIN-1:0];
            else                        xq <= rq_val[W_CIN-1:0];

            if (in_valid) hcnt <= in_last ? {LOG2_H{1'b0}} : (hcnt + 1'b1);
        end
    end

    reg signed [W_ACC-1:0] acc [0:K-1];
    reg                    mac_done;

    genvar c;
    generate
        for (c = 0; c < K; c = c + 1) begin : g_mac
            wire signed [W_CLS-1:0] w_c = $signed(w_row[c*W_CLS +: W_CLS]);
            wire signed [W_CLS+W_CIN-1:0] prod = w_c * xq;
            wire signed [W_ACC-1:0] prod_ext =
                {{(W_ACC-W_CLS-W_CIN){prod[W_CLS+W_CIN-1]}}, prod};

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc[c] <= {W_ACC{1'b0}};
                else if (xq_valid)

                    acc[c] <= (hcnt == 1) ? (b_rom[c] + prod_ext)
                                          : (acc[c]  + prod_ext);
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) mac_done <= 1'b0;
        else        mac_done <= xq_valid && xq_last;
    end

    reg                    emit;
    reg [LOG2_K-1:0]       ecnt;
    reg signed [W_ACC-1:0] best_val;
    reg [LOG2_K-1:0]       best_idx;

    wire signed [W_ACC-1:0] cur = acc[ecnt];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            emit        <= 1'b0;
            ecnt        <= {LOG2_K{1'b0}};
            out_valid   <= 1'b0;
            out_idx     <= {LOG2_K{1'b0}};
            out_logit   <= {W_ACC{1'b0}};
            out_last    <= 1'b0;
            class_valid <= 1'b0;
            class_idx   <= {LOG2_K{1'b0}};
            best_val    <= {W_ACC{1'b0}};
            best_idx    <= {LOG2_K{1'b0}};
        end else begin
            out_valid   <= 1'b0;
            out_last    <= 1'b0;
            class_valid <= 1'b0;

            if (!emit) begin
                if (mac_done) begin
                    emit     <= 1'b1;
                    ecnt     <= {LOG2_K{1'b0}};
                    best_val <= {1'b1, {(W_ACC-1){1'b0}}};
                    best_idx <= {LOG2_K{1'b0}};
                end
            end else begin
                out_valid <= 1'b1;
                out_idx   <= ecnt;
                out_logit <= cur;
                out_last  <= (ecnt == K-1);

                if (cur > best_val) begin
                    best_val <= cur;
                    best_idx <= ecnt;
                end

                if (ecnt == K-1) begin
                    emit        <= 1'b0;
                    class_valid <= 1'b1;
                    class_idx   <= (cur > best_val) ? ecnt : best_idx;
                end else begin
                    ecnt <= ecnt + 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
