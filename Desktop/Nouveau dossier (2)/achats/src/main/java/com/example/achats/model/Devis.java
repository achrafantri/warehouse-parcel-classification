package com.example.achats.model;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDate;

@Entity
@Data
public class Devis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String idProFormat;
    private String nProFormat;
    private LocalDate date;
    private String fournisseur;
    private String matriculeFiscale;
    private String libelle;
    private Double qte;
    private Double montantHT;
    private Double tvaDeductible;
    private Double droitDeTimbre;
    private Double remise;
    private Double montantTTC;
    @OneToOne
    @JoinColumn(name = "id_achat", nullable = false)
    private Achat achat;
    // Getters and Setters

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Achat getAchat() {
        return achat;
    }

    public void setAchat(Achat achat) {
        this.achat = achat;
    }

    public String getIdProFormat() {
        return idProFormat;
    }

    public void setIdProFormat(String idProFormat) {
        this.idProFormat = idProFormat;
    }

    public String getNProFormat() {
        return nProFormat;
    }

    public void setNProFormat(String nProFormat) {
        this.nProFormat = nProFormat;
    }

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
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

    public Double getQte() {
        return qte;
    }

    public void setQte(Double qte) {
        this.qte = qte;
    }

    public Double getMontantHT() {
        return montantHT;
    }

    public void setMontantHT(Double montantHT) {
        this.montantHT = montantHT;
    }

    public Double getTvaDeductible() {
        return tvaDeductible;
    }

    public void setTvaDeductible(Double tvaDeductible) {
        this.tvaDeductible = tvaDeductible;
    }

    public Double getDroitDeTimbre() {
        return droitDeTimbre;
    }

    public void setDroitDeTimbre(Double droitDeTimbre) {
        this.droitDeTimbre = droitDeTimbre;
    }

    public Double getRemise() {
        return remise;
    }

    public void setRemise(Double remise) {
        this.remise = remise;
    }

    public Double getMontantTTC() {
        return montantTTC;
    }

    public void setMontantTTC(Double montantTTC) {
        this.montantTTC = montantTTC;
    }
}
