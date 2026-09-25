`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_decoder_axis -- IP a se stante, AXI-STREAM PURO
//
//   S_AXIS : attivazioni W_ACT bit dal core
//            tlast = ultimo canale dell'ultimo timestep (fine sequenza)
//   M_AXIS : pacchetto di risultato, 32 bit
//
// Nessun AXI-Lite. L'argmax e' gia' calcolato in hardware da
// s4d_classifier: il pacchetto porta direttamente la classe.
//
// FORMATO DEL PACCHETTO (M_AXIS, 32 bit per beat)
//
//   STREAM_LOGITS = 0  (default) -> 1 beat, tlast=1
//   STREAM_LOGITS = 1            -> K beat di logit + 1 beat classe
//                                   (tlast solo sull'ultimo)
//
//   beat di logit  : logit[k], intero con segno a 32 bit
//   beat di classe : [31:24] MAGIC (0xC1, per validare l'inquadramento)
//                    [17]    ERR_OVR  nuova immagine prima di aver
//                                     svuotato il pacchetto precedente
//                    [16]    ERR_LEN  contratto tlast violato in ingresso
//                    [3:0]   class_idx
//
// I flag viaggiano NEL pacchetto: senza AXI-Lite non c'e' altro posto
// dove metterli, e cosi' restano associati all'immagine giusta.
//
// Il decoder non applica backpressure in ingresso (tready sempre alto).
//=====================================================================

module s4d_decoder_axis #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer L        = 1024,   // sCIFAR (era 4096/Speech Commands)
    parameter integer LOG2_L   = 10,      // sCIFAR (era 12/Speech Commands)
    parameter integer K        = 10,      // sCIFAR: 10 classi (era 35/Speech Commands)
    parameter integer LOG2_K   = 4,       // ceil(log2(10)) (era 6/Speech Commands)

    parameter integer W_ACT    = 16,
    parameter integer W_CIN    = 8,
    parameter integer W_CLS    = 8,
    parameter integer W_ACC    = 32,

    parameter integer F_ACT    = 11,
    parameter integer RQ_SHIFT = 6,

    parameter integer BN_SH    = 12,
    parameter integer BN_W_W   = 16,
    parameter integer BN_W_B   = 32,
    parameter integer BN_W_ACC = 34,

    parameter         BN_W_FILE = "l6_bn_w.mem",
    parameter         BN_B_FILE = "l6_bn_b.mem",
    parameter         W_FILE    = "cls_w.mem",
    parameter         B_FILE    = "cls_b.mem",

    parameter integer STREAM_LOGITS = 1,
    parameter [7:0]   MAGIC         = 8'hC1
)(
    input  wire                    aclk,
    input  wire                    aresetn,

    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,
    input  wire signed [W_ACT-1:0] s_axis_tdata,
    input  wire                    s_axis_tlast,

    output wire                    m_axis_tvalid,
    input  wire                    m_axis_tready,
    output wire [31:0]             m_axis_tdata,
    output wire                    m_axis_tlast
);

    localparam integer NBEAT = L * H;
    localparam integer CW    = LOG2_L + LOG2_H;

    //=================================================================
    // core del decoder (invariato)
    //=================================================================
    wire                    logit_valid, logit_last;
    wire [LOG2_K-1:0]       logit_idx, class_idx;
    wire signed [W_ACC-1:0] logit_data;
    wire                    class_valid;

    s4d_decoder #(
        .H(H), .LOG2_H(LOG2_H), .LOG2_L(LOG2_L), .K(K), .LOG2_K(LOG2_K),
        .W_ACT(W_ACT), .W_CIN(W_CIN), .W_CLS(W_CLS),
        .W_ACC(W_ACC), .F_OUT(F_ACT), .RQ_SHIFT(RQ_SHIFT),
        .BN_SH(BN_SH),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .BN_W_ACC(BN_W_ACC),
        .BN_W_FILE(BN_W_FILE), .BN_B_FILE(BN_B_FILE),
        .W_FILE(W_FILE), .B_FILE(B_FILE)
    ) u_dec (
        .clk         (aclk),
        .rst_n       (aresetn),
        .in_valid    (s_axis_tvalid),
        .in_ready    (s_axis_tready),
        .in_data     (s_axis_tdata),
        .in_last_seq (s_axis_tlast),
        .logit_valid (logit_valid),
        .logit_idx   (logit_idx),
        .logit_data  (logit_data),
        .logit_last  (logit_last),
        .class_valid (class_valid),
        .class_idx   (class_idx)
    );

    //=================================================================
    // checker del contratto tlast: L*H beat esatti fra due tlast
    //=================================================================
    reg [CW-1:0] bcnt;
    reg          err_len;
    wire         ibeat = s_axis_tvalid & s_axis_tready;

    // Il blocco always del checker e' piu' sotto, dopo le dichiarazioni
    // della FSM di uscita che usa (st, ST_CLS, obeat).

    //=================================================================
    // cattura dei logit (i logit arrivano 1/ciclo, senza backpressure:
    // vanno parcheggiati prima di poterli streammare)
    //=================================================================
    reg signed [31:0] lbuf [0:K-1];
    always @(posedge aclk) if (logit_valid) lbuf[logit_idx] <= logit_data;

    //=================================================================
    // FSM di uscita
    //=================================================================
    localparam [1:0] ST_IDLE = 2'd0,
                     ST_LOG  = 2'd1,
                     ST_CLS  = 2'd2;

    reg [1:0]        st;
    reg [LOG2_K-1:0] ocnt;
    reg [LOG2_K-1:0] cls_q;
    reg              err_ovr;

    reg              o_valid;
    reg [31:0]       o_data;
    reg              o_last;

    assign m_axis_tvalid = o_valid;
    assign m_axis_tdata  = o_data;
    assign m_axis_tlast  = o_last;

    wire obeat = o_valid & m_axis_tready;
    wire [31:0] cls_pkt = {MAGIC, 6'd0, err_ovr, err_len,
                           {(16-LOG2_K){1'b0}}, cls_q};

    //-----------------------------------------------------------------
    // checker del contratto tlast (registri bcnt / err_len dichiarati
    // sopra): conta i beat in ingresso fra due tlast
    //-----------------------------------------------------------------
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            bcnt <= {CW{1'b0}};
            err_len <= 1'b0;
        end else begin
            if (ibeat) begin
                if (s_axis_tlast) begin
                    if (bcnt != NBEAT[CW-1:0] - 1'b1) err_len <= 1'b1;
                    bcnt <= {CW{1'b0}};
                end else begin
                    bcnt <= bcnt + 1'b1;
                end
            end
            // unico punto di clear: err_len ha un solo driver (vedi ST_CLS,
            // che prima lo azzerava anche li' -- due always sullo stesso reg
            // sintetizzavano due FF sulla stessa rete, DRC MDRV-1).
            if (obeat && (st == ST_CLS)) err_len <= 1'b0;
        end
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            st      <= ST_IDLE;
            ocnt    <= {LOG2_K{1'b0}};
            cls_q   <= {LOG2_K{1'b0}};
            err_ovr <= 1'b0;
            o_valid <= 1'b0;
            o_data  <= 32'd0;
            o_last  <= 1'b0;
        end else begin
            if (obeat) begin
                o_valid <= 1'b0;
                o_last  <= 1'b0;
            end

            case (st)
                //------------------------------------------------------
                ST_IDLE: if (class_valid) begin
                    cls_q <= class_idx;
                    if (STREAM_LOGITS != 0) begin
                        st      <= ST_LOG;
                        ocnt    <= {LOG2_K{1'b0}};
                        o_valid <= 1'b1;
                        o_data  <= lbuf[0];
                        o_last  <= 1'b0;
                    end else begin
                        st      <= ST_CLS;
                        o_valid <= 1'b1;
                        o_data  <= {MAGIC, 6'd0, err_ovr, err_len,
                                    {(16-LOG2_K){1'b0}}, class_idx};
                        o_last  <= 1'b1;
                    end
                end

                //------------------------------------------------------
                ST_LOG: begin
                    if (class_valid) err_ovr <= 1'b1;   // immagine nuova troppo presto
                    if (obeat) begin
                        if (ocnt == K-1) begin
                            st      <= ST_CLS;
                            o_valid <= 1'b1;
                            o_data  <= cls_pkt;
                            o_last  <= 1'b1;
                        end else begin
                            ocnt    <= ocnt + 1'b1;
                            o_valid <= 1'b1;
                            o_data  <= lbuf[ocnt + 1'b1];
                            o_last  <= 1'b0;
                        end
                    end
                end

                //------------------------------------------------------
                ST_CLS: begin
                    if (class_valid) err_ovr <= 1'b1;
                    if (obeat) begin
                        st      <= ST_IDLE;
                        // err_len si azzera nel primo always (unico driver)
                        err_ovr <= 1'b0;
                    end
                end

                default: st <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
