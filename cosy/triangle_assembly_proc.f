      SUBROUTINE TRLOAD(IPATH,ISTAT)
*     Explicit native cache load. No interpreter or subprocess calls.
      USE TAN_MOD, ONLY: TAN_LOAD,TAN_CLEAR
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(LMEM=140000000,LVAR=10000000,LDIM=1000)
      INTEGER NTYP(LVAR),NBEG(LVAR),NEND(LVAR),NMAX(LVAR),
     *        NC(LMEM),NDIM(LDIM)
      DOUBLE PRECISION CC(LMEM)
      COMMON NTYP,NBEG,NEND,NMAX,CC,NC,NDIM,IDIM,IVAR,IMEM
      COMMON /TYID/ NRE,NST,NLO,NCM,NVE,NDA,NCD,NGR
      INTEGER IPATH,ISTAT,RC,J,L
      CHARACTER*4096 PATH
*     Every load attempt invalidates both numerical and exact geometry,
*     including arguments rejected here before the native loader runs.
      CALL TAN_CLEAR()
      RC=1001
      IF(NTYP(IPATH).NE.NST) GOTO 900
      L=NEND(IPATH)-NBEG(IPATH)+1
      IF(L.LT.1.OR.L.GT.LEN(PATH)) GOTO 900
      DO J=1,L
         IF(NC(NBEG(IPATH)+J-1).EQ.0) GOTO 900
         PATH(J:J)=CHAR(NC(NBEG(IPATH)+J-1))
      ENDDO
      CALL TAN_LOAD(PATH(:L),RC)
  900 NTYP(ISTAT)=NRE
      NEND(ISTAT)=NBEG(ISTAT)
      CC(NBEG(ISTAT))=DBLE(RC)
      END

      SUBROUTINE TRFLD(IX,IY,IZ,IBX,IBY,IBZ,IMODE,ISTAT)
*     Radius one gives Taylor coefficients; apply no extra factorials.
*     Modes zero/two accept real coordinates. Mode one accepts mixed
*     real/DA coordinates, composing in the current COSY DA algebra.
*     Outputs must be distinct, including status. On error, field outputs
*     are unchanged. Input/output aliases are safe after center capture.
      USE TAN_MOD, ONLY: TAN_EVAL
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(LMEM=140000000,LVAR=10000000,LDIM=1000)
      INTEGER NTYP(LVAR),NBEG(LVAR),NEND(LVAR),NMAX(LVAR),
     *        NC(LMEM),NDIM(LDIM)
      DOUBLE PRECISION CC(LMEM)
      COMMON NTYP,NBEG,NEND,NMAX,CC,NC,NDIM,IDIM,IVAR,IMEM
      COMMON /TYID/ NRE,NST,NLO,NCM,NVE,NDA,NCD,NGR
      PARAMETER(LEA=100000,LIA=1400000,LNO=99,LNV=40)
      INTEGER IE1(LEA),IE2(LEA),IEO(LEA),IA1(0:LIA),IA2(0:LIA),
     *        NCFLT(LEA),IEW(LNV),IED(LNV),LEW,LEWI,IESP,
     *        NOMAX,NVMAX,NMMAX,NOCUT,LFLT,NFLT
      DOUBLE PRECISION CDA(2*LEA),EPS,EPSMAC
      COMMON /DACOM/ CDA,EPS,EPSMAC,IE1,IE2,IEO,IA1,IA2,NCFLT,
     *       IEW,IED,LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,LFLT,NFLT
      INTEGER IX,IY,IZ,IBX,IBY,IBZ,IMODE,ISTAT
      INTEGER INS(3),OUTS(3),RC,MODE,ORDER,J,K,A,D,EX,EY,EZ
      INTEGER SCR(27),PW(0:6,3),NSIZ,TERM,TEMP,ACC(3),OFF
      DOUBLE PRECISION POINT(3),COEFF(3,84),MV
      LOGICAL ISDA
      INS=(/IX,IY,IZ/)
      OUTS=(/IBX,IBY,IBZ/)
      RC=1004
      DO J=1,3
         IF(OUTS(J).EQ.ISTAT) GOTO 900
         DO K=1,J-1
            IF(OUTS(J).EQ.OUTS(K)) GOTO 900
         ENDDO
      ENDDO
      RC=1002
      IF(NTYP(IMODE).NE.NRE) GOTO 900
      MV=CC(NBEG(IMODE))
      IF(.NOT.IEEE_IS_FINITE(MV)) GOTO 900
      IF(MV.NE.0.D0.AND.MV.NE.1.D0.AND.MV.NE.2.D0) GOTO 900
      MODE=INT(MV)
      RC=1001
      ISDA=.FALSE.
      DO J=1,3
         K=INS(J)
         POINT(J)=0.D0
         IF(NTYP(K).EQ.NRE) THEN
            POINT(J)=CC(NBEG(K))
         ELSEIF(NTYP(K).EQ.NDA) THEN
            ISDA=.TRUE.
            IF(NEND(K).GE.NBEG(K)) THEN
               IF(NC(NBEG(K)).EQ.1) POINT(J)=CC(NBEG(K))
            ENDIF
            DO A=NBEG(K),NEND(K)
               IF(.NOT.IEEE_IS_FINITE(CC(A))) GOTO 900
            ENDDO
         ELSE
            GOTO 900
         ENDIF
         IF(.NOT.IEEE_IS_FINITE(POINT(J))) GOTO 900
      ENDDO
      ORDER=0
      IF(ISDA) THEN
         RC=1002
         IF(MODE.NE.1) GOTO 900
         ORDER=NOMAX
         RC=1003
         IF(ORDER.LT.1.OR.ORDER.GT.6) GOTO 900
      ENDIF
      CALL TAN_EVAL(POINT,MODE,ORDER,1.D0,COEFF,RC)
      IF(RC.NE.0) GOTO 900
      RC=1005
      IF(.NOT.ALL(IEEE_IS_FINITE(COEFF))) GOTO 900
      IF(.NOT.ISDA) THEN
         DO J=1,3
            NTYP(OUTS(J))=NRE
            NEND(OUTS(J))=NBEG(OUTS(J))
            CC(NBEG(OUTS(J)))=COEFF(J,1)
         ENDDO
         RC=0
         GOTO 900
      ENDIF
*     21 powers including three units, two work, three sums, one offset.
      NSIZ=2*NMMAX+2*NOMAX
      CALL FOXALL(SCR,27,NSIZ)
      PW=RESHAPE(SCR(1:21),(/7,3/))
      TERM=SCR(22)
      TEMP=SCR(23)
      ACC=SCR(24:26)
      OFF=SCR(27)
      DO A=1,3
         CALL DACON(PW(0,A),1.D0)
         CALL DACON(OFF,-POINT(A))
         IF(NTYP(INS(A)).EQ.NDA) THEN
            CALL DALIN(INS(A),1.D0,OFF,1.D0,PW(1,A))
         ELSE
            CALL DACON(PW(1,A),0.D0)
         ENDIF
         DO K=2,ORDER
            CALL DAMDA(PW(K-1,A),PW(1,A),PW(K,A))
         ENDDO
         CALL DACON(ACC(A),COEFF(A,1))
      ENDDO
      J=1
      DO D=1,ORDER
         DO EX=0,D
            DO EY=0,D-EX
               EZ=D-EX-EY
               J=J+1
               CALL DAMDA(PW(EX,1),PW(EY,2),TERM)
               CALL DAMDA(TERM,PW(EZ,3),TEMP)
               DO A=1,3
                  CALL DALIN(ACC(A),1.D0,TEMP,COEFF(A,J),TERM)
                  CALL DACOP(TERM,ACC(A))
               ENDDO
            ENDDO
         ENDDO
      ENDDO
      RC=1005
      DO A=1,3
         DO K=NBEG(ACC(A)),NEND(ACC(A))
            IF(.NOT.IEEE_IS_FINITE(CC(K))) GOTO 800
         ENDDO
      ENDDO
      RC=1006
      DO A=1,3
         IF(NEND(ACC(A))-NBEG(ACC(A)).GT.
     *      NMAX(OUTS(A))-NBEG(OUTS(A))) GOTO 800
      ENDDO
      DO A=1,3
         CALL DACOP(ACC(A),OUTS(A))
      ENDDO
      RC=0
  800 CALL FOXDAL(SCR,27)
  900 NTYP(ISTAT)=NRE
      NEND(ISTAT)=NBEG(ISTAT)
      CC(NBEG(ISTAT))=DBLE(RC)
      END
