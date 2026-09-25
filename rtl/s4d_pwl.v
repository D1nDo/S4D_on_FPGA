`timescale 1ns / 1ps
`default_nettype none

module s4d_pwl #(
    parameter integer W_IN      = 16,
    parameter integer F_IN      = 11,
    parameter integer W_OUT     = 16,
    parameter integer F_OUT     = 11,
    parameter integer NSEG      = 256,
    parameter integer LOG2_NSEG = 8,
    parameter integer XH        = 3,
    parameter integer MODE_HI   = 1,
    parameter integer Y_LO_Q    = 0,
    parameter integer Y_HI_Q    = 0,
    parameter         LUT_FILE  = "gelu_lut.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data
);

    // Pipeline STALLABILE globalmente: out_ready basso congela tutti e tre
    // gli stadi insieme. Stesso schema di s4d_affine.
    // NOTA: in_ready e' combinatorio da out_ready -> percorso passante.
    // Se il consumatore genera ready in modo combinatorio si allunga la
    // catena: candidato da sorvegliare in timing closure.
    wire en = out_ready;
    assign in_ready = out_ready;

    localparam integer FRACW  = F_IN + XH + 1 - LOG2_NSEG;
    localparam integer XW     = W_IN + 1;
    localparam signed [XW-1:0] OFFS  = {{(XW-XH-1-F_IN){1'b0}}, 1'b1, {(XH+F_IN){1'b0}}};
    localparam signed [XW-1:0] X_LO  = -OFFS;
    localparam signed [XW-1:0] X_HI  =  OFFS;

    reg [2*W_OUT-1:0] lut [0:NSEG-1];
    initial $readmemh(LUT_FILE, lut);

    wire signed [XW-1:0] x   = {in_data[W_IN-1], in_data};
    wire signed [XW-1:0] xs  = x + OFFS;

    reg                      v1, below1, above1;
    reg [LOG2_NSEG-1:0]      idx1;
    reg [FRACW-1:0]          frac1;
    reg signed [W_IN-1:0]    x1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  v1 <= 1'b0;
        else if (en) v1 <= in_valid;
    end

    always @(posedge clk) if (en) begin
        below1 <= (x <  X_LO);
        above1 <= (x >= X_HI);
        idx1   <= xs[FRACW +: LOG2_NSEG];
        frac1  <= xs[FRACW-1:0];
        x1     <= in_data;
    end

    reg [2*W_OUT-1:0]     seg2;
    reg [FRACW-1:0]       frac2;
    reg                   v2, below2, above2;
    reg signed [W_IN-1:0] x2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  v2 <= 1'b0;
        else if (en) v2 <= v1;
    end

    always @(posedge clk) if (en) begin
        seg2   <= lut[idx1];
        frac2  <= frac1;
        below2 <= below1;
        above2 <= above1;
        x2     <= x1;
    end

    wire signed [W_OUT-1:0] base2  = $signed(seg2[0     +: W_OUT]);
    wire signed [W_OUT-1:0] slope2 = $signed(seg2[W_OUT +: W_OUT]);

    localparam integer PW3 = W_OUT + FRACW + 1;

    wire signed [PW3-1:0] prod = slope2 * $signed({1'b0, frac2});
    wire signed [PW3-1:0] rnd  = {{(PW3-1){1'b0}}, 1'b1} <<< (FRACW-1);

    wire signed [PW3-1:0] interp = base2 + ((prod + rnd) >>> FRACW);

    localparam signed [W_OUT-1:0] YLO = Y_LO_Q[W_OUT-1:0];
    localparam signed [W_OUT-1:0] YHI = Y_HI_Q[W_OUT-1:0];
    localparam signed [PW3-1:0] OMAX = {{(PW3-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [PW3-1:0] OMIN = {{(PW3-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
        end else if (en) begin
            out_valid <= v2;

            if (below2)
                out_data <= YLO;
            else if (above2)

                out_data <= (MODE_HI != 0) ? x2[W_OUT-1:0] : YHI;
            else if (interp > OMAX)
                out_data <= OMAX[W_OUT-1:0];
            else if (interp < OMIN)
                out_data <= OMIN[W_OUT-1:0];
            else
                out_data <= interp[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
