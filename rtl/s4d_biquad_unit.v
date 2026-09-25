`timescale 1ns / 1ps
`default_nettype none

module s4d_biquad_unit #(
    parameter integer NCH       = 2,
    parameter integer MODES     = 32,
    parameter integer LOG2_NB   = 6,
    parameter integer LOG2_MOD  = 5,

    parameter integer W_U   = 16,
    parameter integer F_U   = 11,
    parameter integer W_A   = 24,
    parameter integer F_A   = 22,
    parameter integer W_B   = 16,
    parameter integer F_B   = 15,
    parameter integer W_S   = 32,
    parameter integer F_Y   = 19,
    parameter integer W_D   = 16,
    parameter integer F_D   = 14,

    parameter integer W_BSH = 5,
    parameter integer LOG2_CH = 1,
    parameter integer W_ACC = 40
)(
    input  wire                       clk,
    input  wire                       rst_n,

    input  wire                       cfg_we,
    input  wire [LOG2_NB-1:0]         cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]     cfg_data,

    input  wire                       cfgd_we,
    input  wire [LOG2_CH-1:0]         cfgd_addr,
    input  wire signed [W_D-1:0]      cfgd_data,
    input  wire [W_BSH-1:0]           cfgd_bsh,

    input  wire                       start,
    input  wire                       clr,
    input  wire [NCH*W_U-1:0]         u_vec,
    output reg                        busy,

    output reg                        out_valid,
    output reg  signed [W_ACC-1:0]    out_acc,

    output reg                        ovf
);

    localparam integer NB   = NCH*MODES;
    localparam integer PW   = W_A + W_S;

    localparam integer SH_B0 = F_B + F_U - F_Y;
    localparam integer SH_A = F_A;
    localparam integer SH_D = F_D + F_U - F_Y;

    localparam signed [PW-1:0] SMAX = {{(PW-W_S+1){1'b0}}, {(W_S-1){1'b1}}};
    localparam signed [PW-1:0] SMIN = {{(PW-W_S+1){1'b1}}, {(W_S-1){1'b0}}};

    function [W_S:0] satw;
        input signed [PW-1:0] v;
        input [5:0]           s;
        reg   signed [PW-1:0] t;
        begin
            t = (s > 0) ? ((v + (1 <<< (s-1))) >>> s) : v;
            if      (t > SMAX) satw = {1'b1, SMAX[W_S-1:0]};
            else if (t < SMIN) satw = {1'b1, SMIN[W_S-1:0]};
            else               satw = {1'b0, t[W_S-1:0]};
        end
    endfunction

    (* ram_style = "block" *)
    reg [2*W_A-1:0]       coef_a [0:NB-1];
    reg [2*W_B-1:0]       coef_b [0:NB-1];

    reg [2*W_S-1:0]       state [0:NB-1];

    always @(posedge clk)
        if (cfg_we) begin
            coef_a[cfg_addr] <= cfg_data[2*W_A-1:0];
            coef_b[cfg_addr] <= cfg_data[2*W_A+2*W_B-1:2*W_A];
        end

    reg signed [W_D-1:0] dcoef [0:NCH-1];
    reg [W_BSH-1:0]      bshift [0:NCH-1];
    always @(posedge clk)
        if (cfgd_we) begin
            dcoef[cfgd_addr]  <= cfgd_data;
            bshift[cfgd_addr] <= cfgd_bsh;
        end

    reg [LOG2_NB-1:0] cnt;
    reg               run;
    reg [6:0]         vpipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= {LOG2_NB{1'b0}};
            run <= 1'b0;
        end else if (start && !run) begin
            run <= 1'b1;
            cnt <= {LOG2_NB{1'b0}};
        end else if (run) begin
            if (cnt == NB-1) begin
                run <= 1'b0;
                cnt <= {LOG2_NB{1'b0}};
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) vpipe <= 7'b0;
        else        vpipe <= {vpipe[5:0], run};
    end

    reg clr_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                   clr_r <= 1'b1;
        else if (start && !run)       clr_r <= clr;
        else if (run && (cnt == NB-1)) clr_r <= 1'b0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) busy <= 1'b0;
        else        busy <= run || (|vpipe);
    end

    reg [LOG2_NB-1:0]     a_s1, a_s2, a_s3, a_s4;
    reg [2*W_A-1:0]       cfa_s1;
    reg [2*W_B-1:0]       cfb_s1;
    reg [2*W_S-1:0]       st_s1;
    reg signed [W_U-1:0]  u_s1;

    wire [LOG2_NB-LOG2_MOD-1:0] ch_sel = cnt[LOG2_NB-1:LOG2_MOD];
    wire signed [W_U-1:0]       u_now  = u_vec[ch_sel*W_U +: W_U];

    reg signed [W_D-1:0] d_s1;
    reg [W_BSH-1:0]      bsh_s1;

    always @(posedge clk) begin
        cfa_s1 <= coef_a[cnt];
        cfb_s1 <= coef_b[cnt];
        st_s1  <= clr_r ? {(2*W_S){1'b0}} : state[cnt];
        u_s1   <= u_now;
        d_s1   <= dcoef[ch_sel];
        bsh_s1 <= bshift[ch_sel];
        a_s1   <= cnt;
    end

    wire signed [W_A-1:0] a1_s1 = $signed(cfa_s1[0             +: W_A]);
    wire signed [W_A-1:0] a2_s1 = $signed(cfa_s1[W_A           +: W_A]);
    wire signed [W_B-1:0] b0_s1 = $signed(cfb_s1[0             +: W_B]);
    wire signed [W_B-1:0] b1_s1 = $signed(cfb_s1[W_B           +: W_B]);
    wire signed [W_S-1:0] s1_s1 = $signed(st_s1[0             +: W_S]);
    wire signed [W_S-1:0] s2_s1 = $signed(st_s1[W_S           +: W_S]);

    reg signed [W_B+W_U-1:0] pb0_s2, pb1_s2;
    reg signed [W_S-1:0]     s1_s2, s2_s2;
    reg signed [W_A-1:0]     a1_s2, a2_s2;

    reg signed [W_D+W_U-1:0] pd_s2;
    reg [W_BSH-1:0]          bsh_s2;

    always @(posedge clk) begin
        bsh_s2 <= bsh_s1;
        pb0_s2 <= b0_s1 * u_s1;
        pb1_s2 <= b1_s1 * u_s1;
        pd_s2  <= d_s1 * u_s1;
        s1_s2  <= s1_s1;
        s2_s2  <= s2_s1;
        a1_s2  <= a1_s1;
        a2_s2  <= a2_s1;
        a_s2   <= a_s1;
    end

    wire signed [PW-1:0] pb0_ext = {{(PW-W_B-W_U){pb0_s2[W_B+W_U-1]}}, pb0_s2};
    wire signed [PW-1:0] pb1_ext = {{(PW-W_B-W_U){pb1_s2[W_B+W_U-1]}}, pb1_s2};
    wire signed [PW-1:0] s1_ext  = {{(PW-W_S){s1_s2[W_S-1]}}, s1_s2};

    wire signed [PW-1:0] pd_ext  = {{(PW-W_D-W_U){pd_s2[W_D+W_U-1]}}, pd_s2};

    // stadio 3a: shift variabile registrato
    reg signed [PW-1:0]      shl_s3a, pb0_s3a, pb1_s3a, pd_s3a;
    reg signed [W_S-1:0]     s2_s3a;
    reg signed [W_A-1:0]     a1_s3a, a2_s3a;
    reg [W_BSH-1:0]          bsh_s3a;
    reg [LOG2_NB-1:0]        a_s3a;

    always @(posedge clk) begin
        shl_s3a <= s1_ext <<< bsh_s2;
        pb0_s3a <= pb0_ext;
        pb1_s3a <= pb1_ext;
        pd_s3a  <= pd_ext;
        s2_s3a  <= s2_s2;
        a1_s3a  <= a1_s2;
        a2_s3a  <= a2_s2;
        bsh_s3a <= bsh_s2;
        a_s3a   <= a_s2;
    end

    // stadio 3b: somma registrata
    reg signed [PW-1:0]      sum_s3b, pb1_s3b, pd_s3b;
    reg signed [W_S-1:0]     s2_s3b;
    reg signed [W_A-1:0]     a1_s3b, a2_s3b;
    reg [W_BSH-1:0]          bsh_s3b;
    reg [LOG2_NB-1:0]        a_s3b;

    always @(posedge clk) begin
        sum_s3b <= pb0_s3a + shl_s3a;
        pb1_s3b <= pb1_s3a;
        pd_s3b  <= pd_s3a;
        s2_s3b  <= s2_s3a;
        a1_s3b  <= a1_s3a;
        a2_s3b  <= a2_s3a;
        bsh_s3b <= bsh_s3a;
        a_s3b   <= a_s3a;
    end

    // stadio 3: arrotondamento + saturazione
    wire [W_S:0] y_r   = satw(sum_s3b, bsh_s3b);
    wire [W_S:0] pb1_r = satw(pb1_s3b, bsh_s3b);
    wire [W_S:0] du_r  = satw(pd_s3b, SH_D);

    reg signed [W_S-1:0] y_s3, s2_s3, pb1_s3, du_s3;
    reg signed [W_A-1:0] a1_s3, a2_s3;
    reg                  ov_s3;

    always @(posedge clk) begin
        y_s3   <= $signed(y_r[W_S-1:0]);
        pb1_s3 <= $signed(pb1_r[W_S-1:0]);
        du_s3  <= $signed(du_r[W_S-1:0]);
        ov_s3  <= y_r[W_S] | pb1_r[W_S] | du_r[W_S];
        s2_s3  <= s2_s3b;
        a1_s3  <= a1_s3b;
        a2_s3  <= a2_s3b;
        a_s3   <= a_s3b;
    end

    reg signed [PW-1:0] pa1_s4, pa2_s4;
    reg signed [W_S-1:0] s2_s4, pb1_s4;
    reg                  ov_s4;

    always @(posedge clk) begin
        pa1_s4 <= a1_s3 * y_s3;
        pa2_s4 <= a2_s3 * y_s3;
        s2_s4  <= s2_s3;
        pb1_s4 <= pb1_s3;
        ov_s4  <= ov_s3;
        a_s4   <= a_s3;
    end

    wire [W_S:0] a1y_r = satw(pa1_s4, SH_A);
    wire [W_S:0] a2y_r = satw(pa2_s4, SH_A);

    wire signed [W_S-1:0] a1y = $signed(a1y_r[W_S-1:0]);
    wire signed [W_S-1:0] a2y = $signed(a2y_r[W_S-1:0]);

    wire signed [PW-1:0] s1_new_w = {{(PW-W_S){pb1_s4[W_S-1]}}, pb1_s4}
                                  - {{(PW-W_S){a1y[W_S-1]}},    a1y}
                                  + {{(PW-W_S){s2_s4[W_S-1]}},  s2_s4};

    wire [W_S:0] s1_r = satw(s1_new_w, 0);

    always @(posedge clk)
        if (vpipe[5]) state[a_s4] <= {(-a2y), $signed(s1_r[W_S-1:0])};

    wire stage4_ovf = vpipe[5] & (ov_s4 | a1y_r[W_S] | a2y_r[W_S] | s1_r[W_S]);

    wire [LOG2_MOD-1:0] mode_s3  = a_s3[LOG2_MOD-1:0];
    wire                first_s3 = (mode_s3 == 0);
    wire                last_s3  = (mode_s3 == MODES-1);

    reg  signed [W_ACC-1:0] acc;
    wire signed [W_ACC-1:0] y_ext   = {{(W_ACC-W_S){y_s3[W_S-1]}}, y_s3};
    wire signed [W_ACC-1:0] du_ext  = {{(W_ACC-W_S){du_s3[W_S-1]}}, du_s3};

    wire signed [W_ACC-1:0] acc_nxt = (first_s3 ? du_ext : acc) + y_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc       <= {W_ACC{1'b0}};
            out_valid <= 1'b0;
            out_acc   <= {W_ACC{1'b0}};
        end else begin
            out_valid <= 1'b0;
            if (vpipe[4]) begin
                acc <= acc_nxt;
                if (last_s3) begin
                    out_valid <= 1'b1;
                    out_acc   <= acc_nxt;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          ovf <= 1'b0;
        else if (stage4_ovf) ovf <= 1'b1;
    end

endmodule

`default_nettype wire
