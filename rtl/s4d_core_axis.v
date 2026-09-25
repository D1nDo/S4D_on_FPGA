`timescale 1ns / 1ps
`default_nettype none

module s4d_core_axis #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer L        = 1024,   // sCIFAR (era 4096/Speech Commands)
    parameter integer LOG2_L   = 10,      // sCIFAR (era 12/Speech Commands)
    parameter integer NLAYER   = 6,

    parameter integer NUNITS   = 8,
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,
    parameter integer LOG2_NCH = 4,
    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,
    parameter integer NMAC     = 64,   // TEST 2026-09-10: rimesso al valore originale per
                                        // misurare l'utilizzo risorse del solo core (OOC) ora
                                        // che s4d_wrom_axis (la vera causa della ROM che non
                                        // entrava in BRAM) sta per essere tolto in favore di
                                        // DDR+DMA. Se il numero misurato rientra nel budget,
                                        // NMAC puo' restare 64 (velocita' originale, ~39ms/img);
                                        // se no, via di mezzo (32) o tenere 16 (~136ms/img).
    parameter integer LOG2_NPH = 2,    // torna a seguire NMAC=64: TH/NMAC=256/64=4 fasi

    parameter integer W_ACT    = 16,
    parameter integer F_ACT    = 11,
    parameter integer W_U      = 16,
    parameter integer F_U      = 11,
    parameter integer W_A      = 24,
    parameter integer F_A      = 22,
    parameter integer W_B      = 16,
    parameter integer F_B      = 15,
    parameter integer W_S      = 32,
    parameter integer F_Y      = 19,
    parameter integer W_D      = 16,
    parameter integer F_D      = 13,
    parameter integer W_BSH    = 5,
    parameter integer W_BACC   = 40,
    parameter integer W_SSM    = 16,
    parameter integer F_SSM    = 8,       // dalla quantizzazione sCIFAR
                                          // (mem_trained_bn/quant_report.json);
                                          // ricontrollare se si riaddestra/
                                          // riquantizza il modello
    parameter integer W_W      = 12,
    parameter integer F_W      = 11,
    parameter integer W_MACC   = 32,
    parameter integer F_SIGI   = 11,
    parameter integer F_SIGO   = 14,

    parameter integer BN_W_W   = 16,
    parameter integer BN_W_B   = 32,
    parameter integer BN_W_ACC = 34,
    parameter integer FIFO_LOG2 = 10,

    parameter integer AXI_DW   = 64
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    s_axis_enc_tvalid,
    output wire                    s_axis_enc_tready,
    input  wire signed [W_ACT-1:0] s_axis_enc_tdata,
    input  wire                    s_axis_enc_tlast,
    // stream dei pesi: 6 layer consecutivi, in ordine
    input  wire                    s_axis_w_tvalid,
    output wire                    s_axis_w_tready,
    input  wire [AXI_DW-1:0]       s_axis_w_tdata,
    input  wire                    s_axis_w_tlast,

    output wire                    m_axis_out_tvalid,
    input  wire                    m_axis_out_tready,
    output wire signed [W_ACT-1:0] m_axis_out_tdata,
    output wire                    m_axis_out_tlast,

    // Flag di stato, sticky. Erano reg/wire interni: un errore che
    // nessuno puo' leggere non protegge da niente.
    //   [0] err_out_ovr  uscita sovrascritta (ora NON deve mai scattare)
    //   [1] err_enc_len  lunghezza della sequenza in ingresso errata
    //   [2] err_ovw      la scrittura ha sorpassato la lettura nel seqbuf
    //   [3] layer_ovf    overflow aritmetico nel banco di biquad
    //   [4] err_bank     ready rotto fra banco e mixing (sentinella)
    output wire [4:0]              status_err
);

    localparam integer NWORDS = L * H;
    localparam integer AW     = LOG2_L + LOG2_H;

    localparam [AW-1:0] ADDR_LAST = NWORDS - 1;

    localparam [2:0] ST_IDLE = 3'd0,
                     ST_ENC  = 3'd1,
                     ST_LOAD = 3'd2,
                     ST_PASS = 3'd3,
                     ST_TAIL = 3'd4,
                     ST_DONE = 3'd5;
                     
   //(* mark_debug = "true" *) reg [2:0]     st;
   //(* mark_debug = "true" *) reg [2:0]     m;
   
   reg [2:0]     st;
   reg [2:0]     m;
   reg [AW-1:0]  rptr, wptr;
   reg           rd_run;              
   
   //(* mark_debug = "true" *) reg [2:0]     st;
   //(* mark_debug = "true" *) reg [2:0]     m;
   //(* mark_debug = "true" *) reg [AW-1:0]  rptr, wptr;
   //(* mark_debug = "true" *) reg           rd_run;


    
    wire last_pass = (m == NLAYER[2:0] - 3'd1);
    //(* mark_debug = "true" *) wire                     wl_busy, wl_done;
    
    wire                     wl_busy, wl_done;
    reg                      wl_start;
    wire                     cfg_we;
    wire [LOG2_NU-1:0]       cfg_unit;
    wire [LOG2_NB-1:0]       cfg_addr;
    wire [2*W_A+2*W_B-1:0]   cfg_data;
    wire                     cfgd_we;
    wire [LOG2_H-1:0]        cfgd_ch;
    wire signed [W_D-1:0]    cfgd_data;
    wire [W_BSH-1:0]         cfgd_bsh;
    wire                     ld_bn_we_sh;
    wire [5:0]               ld_bn_sh;
    wire                     ld_bn_we_w, ld_bn_we_b;
    wire [LOG2_H-1:0]        ld_bn_addr;
    wire signed [BN_W_W-1:0] ld_bn_w;
    wire signed [BN_W_B-1:0] ld_bn_b;
    wire                     ld_wo_we, ld_bo_we;
    wire [LOG2_NPH+LOG2_H-1:0] ld_wo_addr;
    wire [3:0]               ld_wo_sel;
    wire [AXI_DW-1:0]        ld_wo_data;
    wire [LOG2_H:0]          ld_bo_addr;
    wire signed [W_MACC-1:0] ld_bo_data;

    s4d_wload_axis #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .LOG2_NB(LOG2_NB), .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_A(W_A), .W_B(W_B), .W_D(W_D), .W_BSH(W_BSH),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .W_MACC(W_MACC), .W_W(W_W), .AXI_DW(AXI_DW)
    ) u_wload (
        .clk(clk), .rst_n(rst_n),
        .start(wl_start),
        .busy(wl_busy), .done(wl_done),.s_axis_w_tvalid(s_axis_w_tvalid),
        .s_axis_w_tready(s_axis_w_tready),
        .s_axis_w_tdata (s_axis_w_tdata),
        .s_axis_w_tlast (s_axis_w_tlast),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data),
        .cfgd_bsh(cfgd_bsh),
        .ld_bn_we_sh(ld_bn_we_sh), .ld_bn_sh(ld_bn_sh),
        .ld_bn_we_w(ld_bn_we_w), .ld_bn_we_b(ld_bn_we_b),
        .ld_bn_addr(ld_bn_addr), .ld_bn_w(ld_bn_w), .ld_bn_b(ld_bn_b),
        .ld_wo_we(ld_wo_we), .ld_wo_addr(ld_wo_addr), .ld_wo_sel(ld_wo_sel),
        .ld_wo_data(ld_wo_data),
        .ld_bo_we(ld_bo_we), .ld_bo_addr(ld_bo_addr), .ld_bo_data(ld_bo_data)
    );

    wire en = 1'b1;                 // il core parte quando arrivano i dati
    reg  img_done;                  // solo uso interno
    //(* mark_debug = "true" *) wire layer_ovf, err_bank, err_ovw;
    wire layer_ovf, err_bank, err_ovw;
    wire lay_out_ready;
    reg  err_out_ovr, err_enc_len;

    wire enc_beat = s_axis_enc_tvalid && s_axis_enc_tready;
    assign s_axis_enc_tready = (st == ST_ENC);

    wire                    buf_wr_en;
    wire [AW-1:0]           buf_wr_addr;
    wire signed [W_ACT-1:0] buf_wr_data;
    wire                    buf_rd_en;
    wire signed [W_ACT-1:0] buf_rd_data;
    
    //(* mark_debug = "true" *) reg                     rd_inflight;
    //(* mark_debug = "true" *) reg                     sk_valid;
    reg                     rd_inflight;
    reg                     sk_valid;
    reg signed [W_ACT-1:0]  sk_data;
    reg                     sk_last;
    reg                     rd_last_q;

    wire lay_in_ready;
    
    //(* mark_debug = "true" *) wire do_rd = rd_run && !rd_inflight && (!sk_valid || lay_in_ready);
    wire do_rd = rd_run && !rd_inflight && (!sk_valid || lay_in_ready);

    assign buf_rd_en = do_rd;

    s4d_seqbuf #(.W(W_ACT), .NWORDS(NWORDS), .AW(AW)) u_buf (
        .clk(clk), .rst_n(rst_n),
        .wr_en(buf_wr_en), .wr_addr(buf_wr_addr), .wr_data(buf_wr_data),
        .rd_en(buf_rd_en), .rd_addr(rptr), .rd_data(buf_rd_data),
        .chk_en((st == ST_PASS) && rd_run), .err_ovw(err_ovw)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_inflight <= 1'b0;
            sk_valid    <= 1'b0;
            sk_last     <= 1'b0;
            rd_last_q   <= 1'b0;
        end else begin
            if (sk_valid && lay_in_ready) sk_valid <= 1'b0;

            if (rd_inflight) begin
                sk_data  <= buf_rd_data;
                sk_last  <= rd_last_q;
                sk_valid <= 1'b1;
            end

            rd_inflight <= do_rd;
            if (do_rd) rd_last_q <= (rptr == ADDR_LAST);

            if (st != ST_PASS) begin
                rd_inflight <= 1'b0;
                sk_valid    <= 1'b0;
            end
        end
    end

    wire                    lay_valid;
    wire signed [W_ACT-1:0] lay_data;
    wire                    lay_last_seq;

    s4d_layer #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .MODES(MODES), .LOG2_MOD(LOG2_MOD), .NCH(NCH), .LOG2_NCH(LOG2_NCH),
        .LOG2_NB(LOG2_NB), .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_ACT(W_ACT), .F_ACT(F_ACT),
        .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A), .W_B(W_B), .F_B(F_B),
        .W_S(W_S), .F_Y(F_Y), .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH),
        .W_BACC(W_BACC), .W_SSM(W_SSM), .F_SSM(F_SSM),
        .W_W(W_W), .F_W(F_W), .W_MACC(W_MACC),
        .F_SIGI(F_SIGI), .F_SIGO(F_SIGO),
        .FIFO_LOG2(FIFO_LOG2),
        .LOADABLE(1), .LD_W(AXI_DW), .BN_SH(0),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .BN_W_ACC(BN_W_ACC),
        .GELU_FILE("gelu_lut.mem"),
        .SIG_FILE ("sigmoid_lut.mem")
    ) u_layer (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data),
        .cfgd_bsh(cfgd_bsh),
        .ld_bn_we_sh(ld_bn_we_sh), .ld_bn_sh(ld_bn_sh),
        .ld_bn_we_w(ld_bn_we_w), .ld_bn_we_b(ld_bn_we_b),
        .ld_bn_addr(ld_bn_addr), .ld_bn_w(ld_bn_w), .ld_bn_b(ld_bn_b),
        .ld_wo_we(ld_wo_we), .ld_wo_addr(ld_wo_addr), .ld_wo_sel(ld_wo_sel),
        .ld_wo_data(ld_wo_data),
        .ld_bo_we(ld_bo_we), .ld_bo_addr(ld_bo_addr), .ld_bo_data(ld_bo_data),
        .in_valid(sk_valid), .in_ready(lay_in_ready), .in_data(sk_data),
        .in_last_seq(sk_valid && sk_last),
        .out_ready(lay_out_ready),
        .out_valid(lay_valid), .out_data(lay_data),
        .out_last_ch(), .out_last_seq(lay_last_seq),
        .ovf(layer_ovf), .err_bank_ovr(err_bank)
    );

    assign buf_wr_en   = (st == ST_ENC)  ? enc_beat
                       : (st == ST_PASS) ? (lay_valid && !last_pass)
                       : 1'b0;
    assign buf_wr_addr = wptr;
    assign buf_wr_data = (st == ST_ENC) ? s_axis_enc_tdata : lay_data;
    
    //(* mark_debug = "true" *) wire dec_fire = lay_valid && last_pass && (st == ST_PASS);
    wire dec_fire = lay_valid && last_pass && (st == ST_PASS);

    // ------------------------------------------------------------------
    // SKID BUFFER di uscita (2 posizioni).
    //
    // Sostituisce il registro singolo, che sovrascriveva il dato quando
    // il consumatore stallava (perdita silenziosa, segnalata solo da
    // err_out_ovr).
    //
    // sk_ready e' REGISTRATO (= !skid_v), non combinatorio da tready:
    // evita di allungare il percorso critico dal DMA fino al layer.
    // Throughput pieno quando non c'e' stallo: con skid_v=0 il beat va
    // direttamente in uscita, 1 per ciclo.
    // ------------------------------------------------------------------
   
    //(* mark_debug = "true" *) reg                    out_v,  skid_v;    
    reg                    out_v,  skid_v;
    reg signed [W_ACT-1:0] out_d,  skid_d;
    reg                    out_l,  skid_l;

    wire sk_out_ready = !skid_v;

    // Il layer va stallato SOLO nell'ultimo pass: negli altri l'uscita
    // finisce nel seqbuf, che accetta sempre.
    assign lay_out_ready = last_pass ? sk_out_ready : 1'b1;

    assign m_axis_out_tvalid = out_v;
    assign m_axis_out_tdata  = out_d;
    assign m_axis_out_tlast  = out_l;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_v       <= 1'b0;
            out_d       <= {W_ACT{1'b0}};
            out_l       <= 1'b0;
            skid_v      <= 1'b0;
            skid_d      <= {W_ACT{1'b0}};
            skid_l      <= 1'b0;
            err_out_ovr <= 1'b0;
            err_enc_len <= 1'b0;
        end else begin
            if (!out_v || m_axis_out_tready) begin
                // lo stadio di uscita si libera: pesca dallo skid se pieno,
                // altrimenti direttamente dall'ingresso
                if (skid_v) begin
                    out_v  <= 1'b1;
                    out_d  <= skid_d;
                    out_l  <= skid_l;
                    skid_v <= 1'b0;
                end else begin
                    out_v  <= dec_fire;
                    out_d  <= lay_data;
                    out_l  <= lay_last_seq;
                end
            end else if (dec_fire && !skid_v) begin
                // uscita bloccata: il beat entrante va nello skid
                skid_v <= 1'b1;
                skid_d <= lay_data;
                skid_l <= lay_last_seq;
            end

            // Sentinella di sovrascrittura.
            // ATTENZIONE: dec_fire e' un VALID, non un fire: il layer
            // tiene alto out_valid per tutta la durata dello stallo, come
            // vuole AXI-Stream. Il beat viene davvero ACCETTATO solo con
            // lay_out_ready alto, quindi la condizione di perdita e'
            // "accettato mentre entrambi gli stadi sono pieni".
            // Con lay_out_ready = !skid_v questo e' strutturalmente
            // impossibile: se scatta, qualcuno ha rotto il ready.
            if (dec_fire && lay_out_ready && skid_v
                         && out_v && !m_axis_out_tready)
                err_out_ovr <= 1'b1;

            if (enc_beat && (s_axis_enc_tlast != (wptr == ADDR_LAST)))
                err_enc_len <= 1'b1;
        end
    end

    assign status_err = {err_bank, layer_ovf, err_ovw,
                         err_enc_len, err_out_ovr};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st       <= ST_IDLE;
            m        <= 3'd0;
            rptr     <= {AW{1'b0}};
            wptr     <= {AW{1'b0}};
            rd_run   <= 1'b0;
            wl_start <= 1'b0;
            img_done <= 1'b0;
        end else begin
            wl_start <= 1'b0;
            img_done <= 1'b0;

            case (st)

                ST_IDLE: begin
                    wptr <= {AW{1'b0}};
                    m    <= 3'd0;
                    if (en && s_axis_enc_tvalid) st <= ST_ENC;
                end

                ST_ENC: begin
                    if (enc_beat) begin
                        if (wptr == ADDR_LAST) begin
                            wptr     <= {AW{1'b0}};
                            wl_start <= 1'b1;
                            st       <= ST_LOAD;
                        end else begin
                            wptr <= wptr + 1'b1;
                        end
                    end
                end

                ST_LOAD: begin
                    if (wl_done) begin
                        rptr   <= {AW{1'b0}};
                        wptr   <= {AW{1'b0}};
                        rd_run <= 1'b1;
                        st     <= ST_PASS;
                    end
                end

                ST_PASS: begin
                    if (do_rd) begin
                        if (rptr == ADDR_LAST) rd_run <= 1'b0;
                        else                   rptr <= rptr + 1'b1;
                    end

                    if (lay_valid && !last_pass) begin
                        if (wptr != ADDR_LAST) wptr <= wptr + 1'b1;
                    end

                    // Si esce dal pass quando l'ultimo campione e' stato
                    // ACCETTATO, non appena presentato. lay_last_seq e' un
                    // valid: nell'ultimo pass il layer lo tiene alto finche'
                    // lo skid di uscita e' pieno (lay_out_ready basso), e in
                    // ST_TAIL dec_fire vale 0, quindi il campione con tlast
                    // deve essere gia' entrato nello stadio di uscita.
                    // Negli altri pass lay_out_ready vale sempre 1.
                    if (lay_last_seq && lay_out_ready) begin
                        if (last_pass) begin
                            st <= ST_TAIL;
                        end else begin
                            m        <= m + 3'd1;
                            wl_start <= 1'b1;
                            st       <= ST_LOAD;
                        end
                    end
                end

                ST_TAIL: if (!out_v) st <= ST_DONE;

                ST_DONE: begin
                    img_done <= 1'b1;
                    st       <= ST_IDLE;
                end

                default: st <= ST_IDLE;

            endcase
        end
    end

endmodule

`default_nettype wire
