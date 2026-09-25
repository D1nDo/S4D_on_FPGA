`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_encoder_ecg_axis -- encoder lineare NIN=12 derivazioni -> H canali
//
//   y[t][ch] = sat16( round( sum_{c<12} W[ch][c]*x[t][c] + b[ch] ) )
//
//   S_AXIS : 32 bit = 2 campioni signed Q5.11 (16 bit ciascuno)
//            [15:0]  = campione pari  (c0, c2, ...)
//            [31:16] = campione dispari (c1, c3, ...)
//            6 parole per timestep, campioni in ordine c0..c11.
//            tlast = ultima parola dell'ultimo timestep.
//   M_AXIS : W_ACT bit, H beat (canale 0..H-1) per ogni timestep,
//            tlast sull'ultimo canale dell'ultimo timestep.
//
//   Buffer A : raccoglie le 6 parole (12 campioni) del timestep
//              successivo mentre B e' in uso.
//   Buffer B : tiene il timestep in corso per i suoi H beat.
//   Trasferimento A -> B : quando B sta emettendo l'ultimo canale
//              (H-esimo beat) e A e' pieno, oppure B e' vuoto.
//
//   Pipeline (stallo globale da m_axis_tready):
//     S1 lettura ROM pesi/bias   S2 12 prodotti in parallelo
//     S3 4 somme parziali (3+3+3+3)   S4 somma totale
//     S5 + bias                  S6 arrotondamento, shift, saturazione
//
//   F_WENC = bit frazionari dell'accumulatore (pesi + ingresso),
//   SH_OUT = F_WENC - F_ACT.  enc_b.mem e' nella stessa scala di F_WENC.
//   enc_w.mem : H righe da NIN*W_WENC bit.  C0_AT_LSB=1 -> il peso del
//   canale c0 e' nei W_WENC bit meno significativi della riga.
//=====================================================================

module s4d_encoder_ecg_axis #(
    parameter integer H          = 128,
    parameter integer LOG2H      = 7,
    parameter integer NIN        = 12,

    parameter integer W_WENC     = 18,
    parameter integer F_WENC     = 31,
    parameter integer W_BENC     = 32,
    parameter integer W_ENCACC   = 40,

    parameter integer W_ACT      = 16,
    parameter integer F_ACT      = 11,

    parameter integer C0_AT_LSB  = 1,

    parameter         ENC_W_FILE = "enc_w.mem",
    parameter         ENC_B_FILE = "enc_b.mem"
)(
    input  wire                        aclk,
    input  wire                        aresetn,

    input  wire                        s_axis_tvalid,
    output wire                        s_axis_tready,
    input  wire [31:0]                 s_axis_tdata,
    input  wire                        s_axis_tlast,

    output wire                        m_axis_tvalid,
    input  wire                        m_axis_tready,
    output wire signed [W_ACT-1:0]     m_axis_tdata,
    output wire                        m_axis_tlast,

    output reg                         sat_sticky,
    output reg                         err_frame
);

    localparam integer W_X    = 16;
    localparam integer NW     = NIN / 2;
    localparam integer CW     = $clog2(NW);
    localparam integer PW     = W_WENC + W_X;
    localparam integer SH_OUT = F_WENC - F_ACT;

    initial begin
        if (SH_OUT < 0) begin
            $display("s4d_encoder_ecg_axis: F_WENC (%0d) < F_ACT (%0d)", F_WENC, F_ACT);
            $finish;
        end
        if (NIN != 12) begin
            $display("s4d_encoder_ecg_axis: la struttura degli addendi e' scritta per NIN=12");
            $finish;
        end
    end

    (* rom_style = "distributed" *)
    reg        [NIN*W_WENC-1:0] rom_w [0:H-1];
    (* rom_style = "distributed" *)
    reg signed [W_BENC-1:0]     rom_b [0:H-1];

    initial begin
        $readmemh(ENC_W_FILE, rom_w);
        $readmemh(ENC_B_FILE, rom_b);
    end

    //=================================================================
    // controllo e buffer A / B
    //=================================================================
    wire stall;
    wire go = ~stall;

    reg [NIN*W_X-1:0] a_vec;
    reg [NIN*W_X-1:0] x_b;
    reg [CW-1:0]      wcnt;
    reg               a_full, a_last;
    reg               busy, last_b;
    reg [LOG2H-1:0]   ch;

    wire ch_last = (ch == H-1);
    wire load    = go & a_full & (~busy | ch_last);

    assign s_axis_tready = ~a_full;
    wire   accept        = s_axis_tvalid & ~a_full;

    always @(posedge aclk) begin
        if (accept) a_vec[{wcnt, 5'b00000} +: 32] <= s_axis_tdata;
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            wcnt      <= {CW{1'b0}};
            a_full    <= 1'b0;
            a_last    <= 1'b0;
            err_frame <= 1'b0;
        end else begin
            if (load) a_full <= 1'b0;
            if (accept) begin
                if (wcnt == NW-1) begin
                    wcnt   <= {CW{1'b0}};
                    a_full <= 1'b1;
                    a_last <= s_axis_tlast;
                end else begin
                    wcnt <= wcnt + 1'b1;
                    if (s_axis_tlast) err_frame <= 1'b1;
                end
            end
        end
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            busy <= 1'b0;
            ch   <= {LOG2H{1'b0}};
        end else if (go) begin
            if (load) begin
                busy <= 1'b1;
                ch   <= {LOG2H{1'b0}};
            end else if (busy) begin
                if (ch_last) busy <= 1'b0;
                else         ch   <= ch + 1'b1;
            end
        end
    end

    always @(posedge aclk) begin
        if (load) begin
            x_b    <= a_vec;
            last_b <= a_last;
        end
    end

    //=================================================================
    // S1: ROM pesi/bias, copia di x, flag valid/last
    //=================================================================
    reg        [NIN*W_WENC-1:0] w_1;
    reg        [NIN*W_X-1:0]    x_1;
    reg signed [W_BENC-1:0]     b_1, b_2, b_3, b_4;
    reg [6:1]                   v_p, l_p;

    always @(posedge aclk) begin
        if (go) begin
            w_1 <= rom_w[ch];
            b_1 <= rom_b[ch];
            x_1 <= x_b;
            b_2 <= b_1;
            b_3 <= b_2;
            b_4 <= b_3;
        end
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            v_p <= 6'b0;
            l_p <= 6'b0;
        end else if (go) begin
            v_p <= {v_p[5:1], busy};
            l_p <= {l_p[5:1], busy & last_b & ch_last};
        end
    end

    //=================================================================
    // S2: 12 moltiplicatori in parallelo
    //=================================================================
    wire [NIN*PW-1:0] p_bus;

    genvar c;
    generate
        for (c = 0; c < NIN; c = c + 1) begin : g_mul
            localparam integer WI = C0_AT_LSB ? c : (NIN - 1 - c);
            wire signed [W_WENC-1:0] wc = w_1[WI*W_WENC +: W_WENC];
            wire signed [W_X-1:0]    xc = x_1[c*W_X +: W_X];
            reg  signed [PW-1:0]     p;
            always @(posedge aclk) begin
                if (go) p <= wc * xc;
            end
            assign p_bus[c*PW +: PW] = p;
        end
    endgenerate

    //=================================================================
    // S3: 4 somme parziali da 3 prodotti
    //=================================================================
    wire [4*W_ENCACC-1:0] ps_bus;

    genvar g;
    generate
        for (g = 0; g < 4; g = g + 1) begin : g_ps
            wire signed [PW-1:0]       q0 = p_bus[(3*g+0)*PW +: PW];
            wire signed [PW-1:0]       q1 = p_bus[(3*g+1)*PW +: PW];
            wire signed [PW-1:0]       q2 = p_bus[(3*g+2)*PW +: PW];
            reg  signed [W_ENCACC-1:0] ps;
            always @(posedge aclk) begin
                if (go) ps <= q0 + q1 + q2;
            end
            assign ps_bus[g*W_ENCACC +: W_ENCACC] = ps;
        end
    endgenerate

    //=================================================================
    // S4: somma totale   S5: + bias
    //=================================================================
    wire signed [W_ENCACC-1:0] s0 = ps_bus[0*W_ENCACC +: W_ENCACC];
    wire signed [W_ENCACC-1:0] s1 = ps_bus[1*W_ENCACC +: W_ENCACC];
    wire signed [W_ENCACC-1:0] s2 = ps_bus[2*W_ENCACC +: W_ENCACC];
    wire signed [W_ENCACC-1:0] s3 = ps_bus[3*W_ENCACC +: W_ENCACC];

    reg signed [W_ENCACC-1:0] sum_4, acc_5;

    always @(posedge aclk) begin
        if (go) begin
            sum_4 <= s0 + s1 + s2 + s3;
            acc_5 <= sum_4 + b_4;
        end
    end

    //=================================================================
    // S6: arrotondamento, shift, saturazione
    //=================================================================
    localparam signed [W_ENCACC-1:0] HI  =  (1 <<< (W_ACT-1)) - 1;
    localparam signed [W_ENCACC-1:0] LO  = -(1 <<< (W_ACT-1));
    localparam signed [W_ENCACC-1:0] RND = (SH_OUT == 0) ? 0 : (1 <<< (SH_OUT-1));

    wire signed [W_ENCACC-1:0] rnd = (acc_5 + RND) >>> SH_OUT;
    wire                       ovf = (rnd > HI) | (rnd < LO);
    wire signed [W_ACT-1:0]    clip = (rnd > HI) ? HI[W_ACT-1:0]
                                    : (rnd < LO) ? LO[W_ACT-1:0]
                                    :              rnd[W_ACT-1:0];

    reg signed [W_ACT-1:0] y_6;

    always @(posedge aclk) begin
        if (go) y_6 <= clip;
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)                    sat_sticky <= 1'b0;
        else if (go & v_p[5] & ovf)      sat_sticky <= 1'b1;
    end

    assign stall         = v_p[6] & ~m_axis_tready;
    assign m_axis_tvalid = v_p[6];
    assign m_axis_tdata  = y_6;
    assign m_axis_tlast  = l_p[6];

endmodule

`default_nettype wire
