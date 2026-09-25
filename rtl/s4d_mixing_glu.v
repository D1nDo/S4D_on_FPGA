`timescale 1ns / 1ps
`default_nettype none

module s4d_mixing_glu #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer NMAC     = 64,
    parameter integer LOG2_NPH = 2,
    parameter integer W_IN     = 16,
    parameter integer F_IN     = 11,
    parameter integer W_W      = 12,
    parameter integer F_W      = 11,
    parameter integer W_ACC    = 32,
    parameter integer W_SIG    = 16,
    parameter integer F_SIGI   = 11,
    parameter integer F_SIGO   = 14,
    parameter integer W_A      = 16,
    parameter integer F_A      = 11,
    parameter integer W_OUT    = 16,
    parameter integer F_OUT    = 11,

    parameter integer LOADABLE = 0,

    parameter integer LD_W     = 64,
    // riusato per NLAYER ruoli (vedi s4d_layer.v): default = layer 0, un file
    // vero, non un placeholder inventato -- chi lo istanzia sovrascrive sempre
    parameter         W_FILE   = "l0_s4d_wout.mem",
    parameter         B_FILE   = "l0_s4d_bout.mem",
    parameter         SIG_FILE = "sigmoid_lut.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    ld_we_w,
    input  wire [LOG2_NPH+LOG2_H-1:0] ld_addr_w,
    input  wire [3:0]              ld_sel_w,
    input  wire [LD_W-1:0]         ld_data_w,
    input  wire                    ld_we_b,
    input  wire [LOG2_H:0]         ld_addr_b,
    input  wire signed [W_ACC-1:0] ld_data_b,

    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last,

    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last
);

    // ------------------------------------------------------------------
    // BACKPRESSURE
    // oen congela SOLO il ramo di uscita: ST_WAIT/ST_GLU, gcnt, la catena
    // di allineamento a_d[]/last_d[], il PWL della sigmoide e i registri
    // di uscita.
    //
    // NON congela ST_PH ne' gli accumulatori: e' una fase separata che
    // non produce uscite, e fermarla ridurrebbe il throughput senza
    // motivo. Nemmeno la cattura in ingresso (cap_fire/icnt/wbank).
    //
    // La backpressure si propaga comunque verso l'ingresso da sola: se
    // ST_GLU non avanza, st non torna a ST_IDLE, ph_last non scatta,
    // nfull non decrementa e in_ready va basso dopo due frame.
    // ------------------------------------------------------------------
    wire oen = out_ready;

    localparam integer TH    = 2*H;
    localparam integer NPH   = TH / NMAC;
    localparam integer FACC  = F_W + F_IN;
    localparam integer RQ_G  = FACC - F_SIGI;
    localparam integer RQ_A  = FACC - F_A;
    localparam integer SH_O  = F_A + F_SIGO - F_OUT;

    initial begin
        if (NPH * NMAC != TH) begin
            $display("s4d_mixing_glu: NMAC=%0d non divide TH=%0d", NMAC, TH);
            $finish;
        end
        if ((1 << LOG2_NPH) < NPH) begin
            $display("s4d_mixing_glu: LOG2_NPH=%0d troppo stretto per NPH=%0d",
                     LOG2_NPH, NPH);
            $finish;
        end
    end

    localparam integer NBANK = (NMAC*W_W) / LD_W;

    reg signed [W_ACC-1:0] b_rom [0:TH-1];

    generate
    if (LOADABLE == 0) begin : g_init
        initial $readmemh(B_FILE, b_rom);
    end else begin : g_load
        always @(posedge clk)
            if (ld_we_b) b_rom[ld_addr_b] <= ld_data_b;
    end
    endgenerate

    reg signed [W_IN-1:0] g_buf [0:2*H-1];
    reg                   wbank, rbank;
    reg [LOG2_H-1:0]      icnt;
    reg [1:0]             nfull;

    assign in_ready = (nfull < 2'd2);

    wire cap_fire = in_valid && in_ready;
    wire cap_done = cap_fire && (icnt == H-1);

    always @(posedge clk) if (cap_fire) g_buf[{wbank, icnt}] <= in_data;

// synthesis translate_off
    always @(posedge clk)
        if (cap_fire && (in_last != (icnt == H-1))) begin
            $display("s4d_mixing_glu: in_last a icnt=%0d, atteso a %0d -- il",
                     icnt, H-1);
            $display("  frame in ingresso non e' lungo H: sincronizzazione rotta");
            $finish;
        end
// synthesis translate_on

    localparam [1:0] ST_IDLE = 2'd0,
                     ST_PH   = 2'd1,
                     ST_WAIT = 2'd2,
                     ST_GLU  = 2'd3;

    reg [1:0]           st;
    reg [LOG2_NPH-1:0]  ph;
    reg [LOG2_H-1:0]    hcnt;
    reg [LOG2_H-1:0]    gcnt;

    wire ph_last_h = (st == ST_PH)  && (hcnt == H-1);
    wire ph_last   = ph_last_h && (ph == NPH-1);
    wire glu_last  = (st == ST_GLU) && (gcnt == H-1);
    // fire: un campione entra nel PWL solo quando l'uscita avanza
    wire glu_fire  = (st == ST_GLU) && oen;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wbank <= 1'b0;
            rbank <= 1'b0;
            icnt  <= {LOG2_H{1'b0}};
            nfull <= 2'd0;
            st    <= ST_IDLE;
            ph    <= {LOG2_NPH{1'b0}};
            hcnt  <= {LOG2_H{1'b0}};
            gcnt  <= {LOG2_H{1'b0}};
        end else begin
            if (cap_fire) begin
                if (icnt == H-1) begin
                    icnt  <= {LOG2_H{1'b0}};
                    wbank <= ~wbank;
                end else begin
                    icnt <= icnt + 1'b1;
                end
            end

            case (st)
                ST_IDLE: if (nfull != 2'd0) begin
                    st   <= ST_PH;
                    ph   <= {LOG2_NPH{1'b0}};
                    hcnt <= {LOG2_H{1'b0}};
                end

                ST_PH: begin
                    if (hcnt == H-1) begin
                        hcnt <= {LOG2_H{1'b0}};
                        if (ph == NPH-1) begin
                            st    <= ST_WAIT;
                            gcnt  <= {LOG2_H{1'b0}};
                            rbank <= ~rbank;
                        end else begin
                            ph <= ph + 1'b1;
                        end
                    end else begin
                        hcnt <= hcnt + 1'b1;
                    end
                end

                ST_WAIT: if (oen) st <= ST_GLU;

                ST_GLU: if (oen) begin
                    if (gcnt == H-1) begin
                        gcnt <= {LOG2_H{1'b0}};
                        st   <= ST_IDLE;
                    end else begin
                        gcnt <= gcnt + 1'b1;
                    end
                end

                default: st <= ST_IDLE;
            endcase

            case ({cap_done, ph_last})
                2'b10: nfull <= nfull + 2'd1;
                2'b01: nfull <= nfull - 2'd1;
                default: ;
            endcase
        end
    end

    wire [NMAC*W_W-1:0]   w_row;
    reg signed [W_IN-1:0] x_q;
    reg                   x_valid, x_first;
    reg [LOG2_NPH-1:0]    ph_q;

    wire [31:0] w_addr = ph * H + hcnt;
    wire [LOG2_NPH+LOG2_H-1:0] w_raddr = w_addr[LOG2_NPH+LOG2_H-1:0];

    genvar wb;
    generate
    for (wb = 0; wb < NBANK; wb = wb + 1) begin : g_wbank

        // pesi in Block RAM: 512x64 bit per banco = 1 RAMB36 (512x72).
        // In LUTRAM (default) i 12 banchi costavano ~8200 LUT, cioe' il 15%
        // del chip, e il routing non chiudeva per congestione.
        (* ram_style = "block" *)
        reg [LD_W-1:0] mem [0:NPH*H-1];
        reg [LD_W-1:0] q;

        if (LOADABLE == 0) begin : g_rom

            reg [NMAC*W_W-1:0] w_file [0:NPH*H-1];
            integer fi;
            initial begin
                $readmemh(W_FILE, w_file);
                for (fi = 0; fi < NPH*H; fi = fi + 1)
                    mem[fi] = w_file[fi][wb*LD_W +: LD_W];
            end
        end else begin : g_wr

            localparam [3:0] BSEL = wb;
            always @(posedge clk)
                if (ld_we_w && (ld_sel_w == BSEL))
                    mem[ld_addr_w] <= ld_data_w;
        end

        always @(posedge clk)
            if (rst_n) q <= mem[w_raddr];

        assign w_row[wb*LD_W +: LD_W] = q;

    end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_valid <= 1'b0;
            x_first <= 1'b0;
            ph_q    <= {LOG2_NPH{1'b0}};
        end else begin
            x_q     <= g_buf[{rbank, hcnt}];
            x_valid <= (st == ST_PH);
            x_first <= (st == ST_PH) && (hcnt == 0);
            ph_q    <= ph;
        end
    end

    wire [NMAC*W_ACC-1:0] pext_v;

    genvar m;
    generate
        for (m = 0; m < NMAC; m = m + 1) begin : g_mac
            wire signed [W_W-1:0]      w_m  = $signed(w_row[m*W_W +: W_W]);
            wire signed [W_W+W_IN-1:0] prod = w_m * x_q;
            assign pext_v[m*W_ACC +: W_ACC] =
                {{(W_ACC-W_W-W_IN){prod[W_W+W_IN-1]}}, prod};
        end
    endgenerate

    reg [TH*W_ACC-1:0] acc;

    integer mm;
    reg [LOG2_H:0] c_idx;

    always @(posedge clk) begin
        if (x_valid)
            for (mm = 0; mm < NMAC; mm = mm + 1) begin

                c_idx = ph_q * NMAC + mm;
                acc[c_idx*W_ACC +: W_ACC] <=
                    (x_first ? b_rom[c_idx] : $signed(acc[c_idx*W_ACC +: W_ACC]))
                    + $signed(pext_v[mm*W_ACC +: W_ACC]);
            end
    end

    wire glu = (st == ST_GLU);

    function signed [W_SIG-1:0] rq;
        input signed [W_ACC-1:0] v;
        input integer            s;
        input integer            w;
        reg   signed [W_ACC-1:0] t;
        reg   signed [W_ACC-1:0] hi, lo;
        begin
            t  = (s > 0) ? ((v + (1 <<< (s-1))) >>> s) : (v <<< (-s));
            hi = (1 <<< (w-1)) - 1;
            lo = -(1 <<< (w-1));
            if      (t > hi) rq = hi[W_SIG-1:0];
            else if (t < lo) rq = lo[W_SIG-1:0];
            else             rq = t[W_SIG-1:0];
        end
    endfunction

    wire signed [W_ACC-1:0] acc_a = $signed(acc[gcnt*W_ACC +: W_ACC]);
    wire signed [W_ACC-1:0] acc_g = $signed(acc[(H + gcnt)*W_ACC +: W_ACC]);

    wire signed [W_SIG-1:0] g_q = rq(acc_g, RQ_G, W_SIG);
    wire signed [W_A-1:0]   a_q = rq(acc_a, RQ_A, W_A);

    wire                    sig_valid;
    wire                    sig_in_ready;
    wire signed [W_SIG-1:0] sig_out;

    s4d_pwl #(
        .W_IN(W_SIG), .F_IN(F_SIGI), .W_OUT(W_SIG), .F_OUT(F_SIGO),
        .NSEG(256), .LOG2_NSEG(8), .XH(3),
        .MODE_HI(0), .Y_LO_Q(0), .Y_HI_Q(1 << F_SIGO),
        .LUT_FILE(SIG_FILE)
    ) u_sig (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (glu),
        .in_ready  (sig_in_ready),   // == oen, non serve consultarlo
        .in_data   (g_q),
        .out_ready (oen),
        .out_valid (sig_valid),
        .out_data  (sig_out)
    );

    reg signed [W_A-1:0] a_d [0:2];
    reg [2:0]            last_d;

    // a_d[] e last_d devono avanzare ESATTAMENTE come la pipeline del PWL
    // (3 stadi): stesso enable, altrimenti il prodotto a*sigmoid si
    // disallinea e ottieni un errore che sembra aritmetico.
    always @(posedge clk) if (oen) begin
        a_d[0] <= a_q;
        a_d[1] <= a_d[0];
        a_d[2] <= a_d[1];
        last_d <= {last_d[1:0], glu_last};
    end

    localparam integer PWO = W_A + W_SIG;
    localparam signed [PWO-1:0] OMAX = {{(PWO-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [PWO-1:0] OMIN = {{(PWO-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    wire signed [PWO-1:0] pgl  = a_d[2] * sig_out;
    wire signed [PWO-1:0] prnd = {{(PWO-1){1'b0}}, 1'b1} <<< (SH_O-1);
    wire signed [PWO-1:0] pout = (pgl + prnd) >>> SH_O;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_last  <= 1'b0;
        end else if (oen) begin
            out_valid <= sig_valid;
            out_last  <= sig_valid && last_d[2];

            if      (pout > OMAX) out_data <= OMAX[W_OUT-1:0];
            else if (pout < OMIN) out_data <= OMIN[W_OUT-1:0];
            else                  out_data <= pout[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
