package com.example.achats.model;

import jakarta.persistence.*;
import lombok.Data;

/**
 * sna3na entité produit marbouta bel bdd
 * sta3malna feha lombock bech getter w setter yetsan3ou wa7adhom (@data)
 */
@Entity
@Data
public class Produit {

    /**
     * Id mte3 produit, yet3mal automatiquement.
     */

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String nom;
    private String code;
    private String designation;
    private Double prixUnitaire;
    private String photo;

    @ManyToOne
    @JoinColumn(name = "id_categorie", nullable = false)
    private Categorie categorie;
    private String fournisseur;
    private String numfacture;
    private String numservice;

    /** getteer w setter zednehom manuellement 5ater sar erreur fel intellij ma 9rach lombok*/
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getNom() {
        return nom;
    }

    public void setNom(String nom) {
        this.nom = nom;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public String getDesignation() {
        return designation;
    }

    public void setDesignation(String designation) {
        this.designation = designation;
    }

    public Double getPrixUnitaire() {
        return prixUnitaire;
    }

    public void setPrixUnitaire(Double prixUnitaire) {
        this.prixUnitaire = prixUnitaire;
    }

    public String getPhoto() {
        return photo;
    }

    public void setPhoto(String photo) {
        this.photo = photo;
    }

    public Categorie getCategorie() {
        return categorie;
    }

    public void setCategorie(Categorie categorie) {
        this.categorie = categorie;
    }

    public String getFournisseur() {
        return fournisseur;
    }

    public void setFournisseur(String fournisseur) {
        this.fournisseur = fournisseur;
    }

    public String getNumfacture() {
        return numfacture;
    }

    public void setNumfacture(String numfacture) {
        this.numfacture = numfacture;
    }

    public String getNumservice() {
        return numservice;
    }

    public void setNumservice(String numservice) {
        this.numservice = numservice;
    }
}
