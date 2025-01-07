package com.example.achats.model;

import jakarta.persistence.*;
import java.util.Date;

@Entity
public class BonDeCommande {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idBon;
    private String nBonDeCommande;
    private Date date;
    private String fournisseur;
    private String matriculeFiscale;
    private String libelle;
    private int qte;
    private double montantHT;
    private double tvaDeductible;
    private String raS;
    private double droitDeTimbre;
    private double remise;
    private double montantTTC;
    @OneToOne
    @JoinColumn(name = "id_achat", nullable = false)
    private Achat achat;
    public Achat getAchat() {
        return achat;
    }

    public void setAchat(Achat achat) {
        this.achat = achat;
    }

    // Getters and Setters
    public Long getIdBon() {
        return idBon;
    }

    public void setIdBon(Long idBon) {
        this.idBon = idBon;
    }

    public String getNBonDeCommande() {
        return nBonDeCommande;
    }

    public void setNBonDeCommande(String nBonDeCommande) {
        this.nBonDeCommande = nBonDeCommande;
    }

    public Date getDate() {
        return date;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public String getFournisseur() {
        return fournisseur;
    }

    public void setFournisseur(String fournisseur) {
        this.fournisseur = fournisseur;
    }

    public String getMatriculeFiscale() {
        return matriculeFiscale;
    }

    public void setMatriculeFiscale(String matriculeFiscale) {
        this.matriculeFiscale = matriculeFiscale;
    }

    public String getLibelle() {
        return libelle;
    }

    public void setLibelle(String libelle) {
        this.libelle = libelle;
    }

    public int getQte() {
        return qte;
    }

    public void setQte(int qte) {
        this.qte = qte;
    }

    public double getMontantHT() {
        return montantHT;
    }

    public void setMontantHT(double montantHT) {
        this.montantHT = montantHT;
    }

    public double getTvaDeductible() {
        return tvaDeductible;
    }

    public void setTvaDeductible(double tvaDeductible) {
        this.tvaDeductible = tvaDeductible;
    }

    public String getRaS() {
        return raS;
    }

    public void setRaS(String raS) {
        this.raS = raS;
    }

    public double getDroitDeTimbre() {
        return droitDeTimbre;
    }

    public void setDroitDeTimbre(double droitDeTimbre) {
        this.droitDeTimbre = droitDeTimbre;
    }

    public double getRemise() {
        return remise;
    }

    public void setRemise(double remise) {
        this.remise = remise;
    }

    public double getMontantTTC() {
        return montantTTC;
    }

    public void setMontantTTC(double montantTTC) {
        this.montantTTC = montantTTC;
    }


}
