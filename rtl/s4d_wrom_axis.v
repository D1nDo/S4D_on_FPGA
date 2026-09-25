`timescale 1ns / 1ps
`default_nettype none

// ============================================================================
// s4d_wrom_axis -- buffer singolo (BRAM) fra axi_dma_1 (MM2S, DDR) e s_axis_w
// di s4d_core_axis.
//
// VERSIONE A SINGOLO BUFFER (non ping-pong): sostituisce la precedente
// versione a doppio banco (bufA/bufB) perche' quella, pur corretta dal punto
// di vista logico, non veniva inferita in Block RAM da Vivado (WARNING
// Synth 8-6849 "Infeasible attribute ram_style=block", fallback su LUTRAM,
// ~40k primitive RAM64M/RAM64X1D -- oltre 2 volte il budget di RAM
// distribuita dell'xc7z020) e mandava in errore il DRC di place_design
// (UTLZ-1 su LUT-as-Distributed-RAM/Memory e RAMD64E). Il sospetto e' che
// il mux a runtime fra due array diversi sul lato lettura (bufA/bufB
// selezionati da rd_bank) impedisse a Vivado di riconoscere il pattern
// come semplice simple-dual-port BRAM. Con un solo array quel mux sparisce.
//
// COSTO FUNZIONALE rispetto alla versione a doppio banco: NESSUN overlap
// fra il caricamento del layer N+1 e il consumo del layer N. Il core deve
// aspettare che il buffer si svuoti completamente prima che axi_dma_1
// possa iniziare a caricare il layer successivo (s_axis_tready resta basso
// finche' il buffer non e' stato interamente drenato dal lato lettura).
// Nel caso peggiore: uno stallo per ognuno degli NLAYER layer per ogni
// inferenza, della durata di un intero trasferimento DMA da NBEAT beat.
//
// Nome del modulo e porte m_axis_w_* INVARIATI rispetto alla versione
// precedente apposta: la cella s4d_wrom_axis_0 gia' esistente nel block
// diagram, dopo un refresh dell'IP, si ritrova solo le porte NUOVE (s_axis_*
// in ingresso, err_len in uscita) da collegare -- non serve cancellarla e
// ricrearla.
//
// SINCRONIZZAZIONE -- NESSUN CANALE DI STATO CUSTOM, solo AXI-Stream
// nativo. Il software non ha bisogno di sapere nulla sullo stato interno:
// gli basta accodare i layer in ordine fisso (0,1,2,3,4,5,0,1,...) e
// aspettare l'interrupt di completamento NATIVO di axi_dma_1
// (mm2s_introut, gia' esistente, stesso usato da axi_dma_0) prima di
// accodare il successivo. Se il buffer non e' ancora libero quando
// software tenta il prossimo trasferimento, s_axis_tready resta basso e
// axi_dma_1 si blocca da solo in mid-stream (backpressure AXI-Stream
// normale) finche' non si libera -- nessuna race, nessun caso speciale
// per il priming iniziale.
//
// LATO SCRITTURA (s_axis_*, slave): un solo ingresso stream. Il software
// programma axi_dma_1 per trasferire ESATTAMENTE NBEAT beat (un layer) per
// ogni transazione.
//
// LATO LETTURA (m_axis_w_*, master): stesso contratto ESATTO della vecchia
// ROM verso s4d_core_axis/s_axis_w (stream libero, tlast informativo/
// ignorato dal consumatore -- vedi s4d_wload_axis.v). Se il buffer si
// esaurisce e il caricamento successivo non e' ancora arrivato, il modulo
// si FERMA (tvalid basso) invece di proseguire con dati vecchi o
// sbagliati: nel caso peggiore il core aspetta, non riceve mai un layer
// sbagliato.
//
// err_len : sticky, PURAMENTE diagnostico (non partecipa alla
//           sincronizzazione). Un trasferimento in ingresso e' terminato
//           (tlast) con un conteggio di beat diverso da NBEAT, oppure ha
//           superato NBEAT senza tlast. Per costruzione non puo' succedere
//           se axi_dma_1 e' programmato con la lunghezza giusta -- serve
//           solo ad accorgersi subito se il software sbaglia il
//           descrittore DMA, invece di scoprirlo da una classificazione
//           sbagliata senza nessun indizio del perche'.
// ============================================================================
module s4d_wrom_axis #(
    parameter integer H        = 128,
    parameter integer NUNITS   = 8,
    parameter integer LOG2_NB  = 9,
    parameter integer NMAC     = 16,
    parameter integer W_A      = 24,
    parameter integer W_B      = 16,
    parameter integer W_D      = 16,
    parameter integer W_BSH    = 5,
    parameter integer W_W      = 12,
    parameter integer AXI_DW   = 64
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // lato scrittura: da axi_dma_1 (MM2S), un layer per trasferimento
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,
    input  wire [AXI_DW-1:0]     s_axis_tdata,
    input  wire                  s_axis_tlast,

    // lato lettura: verso s4d_core_axis/s_axis_w -- invariato
    output wire                  m_axis_w_tvalid,
    input  wire                  m_axis_w_tready,
    output wire [AXI_DW-1:0]     m_axis_w_tdata,
    output wire                  m_axis_w_tlast,

    // diagnostica opzionale (vedi commento in testa al file) -- puo'
    // restare scollegata nel block diagram, non serve al funzionamento
    output reg                   err_len
);

    // Stesse formule di s4d_wload_axis.v: se cambiano NUNITS/MODES/NMAC/etc.
    // questo modulo resta in lock-step automaticamente.
    localparam integer NCOEF  = NUNITS * (1 << LOG2_NB);
    localparam integer TH     = 2 * H;
    localparam integer NPH    = TH / NMAC;
    localparam integer NWO    = NPH * H;
    localparam integer WOBEAT = (NMAC * W_W) / AXI_DW;

    localparam integer H_END  = 1;
    localparam integer C_END  = H_END + 2 * NCOEF;
    localparam integer D_END  = C_END + H;
    localparam integer NW_END = D_END + H;
    localparam integer NB_END = NW_END + H;
    localparam integer BO_END = NB_END + TH;
    localparam integer WO_END = BO_END + NWO * WOBEAT;
    localparam integer NBEAT  = WO_END;

    localparam integer AW = $clog2(NBEAT);

    (* ram_style = "block" *) reg [AXI_DW-1:0] buf_mem [0:NBEAT-1];

    reg buf_ready;   // 1 = pieno, pronto per la lettura

    // ------------------------------------------------------------------
    // LATO SCRITTURA
    // ------------------------------------------------------------------
    wire wr_can_start = !buf_ready;   // il buffer e' libero (gia' drenato)

    reg          wr_busy;
    reg [AW-1:0] wr_addr;

    assign s_axis_tready = wr_busy || wr_can_start;

    wire wr_hs        = s_axis_tvalid && s_axis_tready;
    wire wr_last_beat = (wr_addr == NBEAT[AW-1:0] - 1'b1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_busy <= 1'b0;
            wr_addr <= {AW{1'b0}};
            err_len <= 1'b0;   // unico driver di err_len, vedi nota in decoder.v
                                // su cosa succede con due always sullo stesso reg
        end else if (wr_hs) begin
            if (!wr_busy) begin
                wr_busy <= 1'b1;
            end

            if (wr_last_beat) begin
                wr_busy <= 1'b0;
                wr_addr <= {AW{1'b0}};
                if (!s_axis_tlast) err_len <= 1'b1;   // atteso tlast qui, non arrivato
            end else begin
                wr_addr <= wr_addr + 1'b1;
                if (s_axis_tlast) err_len <= 1'b1;    // tlast arrivato troppo presto
            end
        end
    end

    // Scrittura della memoria isolata in un processo sincrono puro (senza
    // reset asincrono): i BRAM Xilinx non hanno controllo asincrono sulla
    // matrice, quindi l'array deve stare da solo in un always @(posedge clk)
    // perche' Vivado possa inferirlo come Block RAM invece di tentare
    // (fallendo) di dissolverlo in singoli flip-flop.
    always @(posedge clk) begin
        if (wr_hs) begin
            buf_mem[wr_addr] <= s_axis_tdata;
        end
    end

    // ------------------------------------------------------------------
    // LATO LETTURA (stesso schema della vecchia ROM)
    // ------------------------------------------------------------------
    reg [AW-1:0]  rd_addr;
    reg           tvalid_r;
    reg [AXI_DW-1:0] data_r;
    reg           last_r;

    wire rd_can_go = buf_ready;
    wire refill    = !tvalid_r || m_axis_w_tready;
    wire at_end    = (rd_addr == NBEAT[AW-1:0] - 1'b1);
    wire rd_hs_last = refill && rd_can_go && at_end;   // usato sotto per svuotare il buffer

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_ready <= 1'b0;
        end else begin
            if (wr_hs && wr_last_beat)
                buf_ready <= 1'b1;
            if (rd_hs_last)
                buf_ready <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_addr  <= {AW{1'b0}};
            tvalid_r <= 1'b0;
            last_r   <= 1'b0;
        end else if (refill) begin
            if (rd_can_go) begin
                last_r   <= at_end;
                tvalid_r <= 1'b1;
                if (at_end) begin
                    rd_addr <= {AW{1'b0}};
                end else begin
                    rd_addr <= rd_addr + 1'b1;
                end
            end else begin
                tvalid_r <= 1'b0;   // buffer non ancora pronto: stallo, mai dati vecchi
            end
        end
    end

    // Lettura della memoria isolata in un processo sincrono puro (senza
    // reset asincrono), stesso motivo della scrittura. Nota: data_r non ha
    // piu' un valore di reset -- e' innocuo perche' tvalid_r (che resta
    // resettato correttamente) maschera sempre data_r finche' non e' stato
    // scritto almeno una volta da una lettura valida.
    always @(posedge clk) begin
        if (refill && rd_can_go) begin
            data_r <= buf_mem[rd_addr];
        end
    end

    assign m_axis_w_tvalid = tvalid_r;
    assign m_axis_w_tdata  = data_r;
    assign m_axis_w_tlast  = tvalid_r && last_r;

endmodule

`default_nettype wire
